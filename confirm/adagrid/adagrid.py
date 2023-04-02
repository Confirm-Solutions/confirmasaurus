import abc
import asyncio
import contextlib
import logging
import queue
import time
from pprint import pformat

import jax
import numpy as np

from .asyncio_runner import Runner
from .db import DatabaseLogging
from .db import DuckDBTiles
from .init import init
from .step import new_step
from .step import process_tiles
from .step import submit_packet_df
from .step import wait_for_packets
from .step import WorkerStatus

logger = logging.getLogger(__name__)


def pass_control_to_backend(algo_type, kwargs):
    backend = kwargs.get("backend", None)
    if backend is None:
        backend = LocalBackend()
    kwargs.update(backend.get_cfg())

    return backend.entrypoint(algo_type, kwargs)


def setup_db(backend, cfg, require_fresh=False):
    if cfg["job_name"] is None:
        if cfg["job_name_prefix"] is None:
            cfg["job_name_prefix"] = "unnamed"
        cfg["job_name"] = f"{cfg['job_name_prefix']}_{time.strftime('%Y%m%d_%H%M%S')}"

    db = backend.connect_to_db(cfg)
    if require_fresh and db.does_table_exist("config"):
        raise ValueError(
            "Table 'config' already exists." " Please choose a fresh job_name."
        )
    return db


def entrypoint(backend, algo_type, kwargs):
    """
    Passing control of the entrypoint to the backend allows executing the
    leader somewhere besides the launching machine.

    The default behavior is to just run the leader locally.
    """

    db = setup_db(backend, kwargs, require_fresh=True)
    with DatabaseLogging(db) as db_logging:
        with Runner() as runner:
            try:
                runner.run(async_entrypoint(backend, db, algo_type, kwargs))
            except Exception as e:  # noqa
                # bare except is okay because we want to log the exception. the
                # re-raise will preserve the original stack trace.
                logging.error("Adagrid error", exc_info=e)
                raise
            finally:
                # This should be the last thing we do so we can be sure that all logging
                # messages have been sent to the database.
                db_logging.flush()
                runner.run(db.finalize())
    return db


async def async_entrypoint(backend, db, algo_type, kwargs):
    entry_time = time.time()
    with timer("init"):
        algo, initial_tiles_df, parent_count = init(db, algo_type, kwargs)
    expected_counts = dict()
    next_step = 1

    start = time.time()
    async with backend.setup(algo):
        logger.debug(f"backend.setup() took {time.time() - start:.2f} seconds")

        with timer("process_initial_incompletes"):
            expected_counts[0] = await wait_for_packets(
                backend,
                algo,
                await submit_packet_df(
                    backend, algo, initial_tiles_df, refine_deepen=False
                ),
            )

        stopping_indicator = 0
        n_parallel_steps = algo.cfg["n_parallel_steps"]
        processing_tasks = queue.Queue()
        verify_tasks = []
        for step_id in range(next_step, algo.cfg["n_steps"]):
            basal_step_id = max(step_id - n_parallel_steps, 0)
            with timer("wait for basal step"):
                await db.wait_for_step(basal_step_id, step_id, expected_counts)

            done_verifies = [t for t in verify_tasks if t is not None and t.done()]
            await asyncio.gather(*done_verifies)
            verify_tasks = [
                t for t in verify_tasks if t is not None and not t.done()
            ] + [db.verify(basal_step_id)]

            if time.time() - entry_time > kwargs["timeout"]:
                logger.info("Job timeout reached, stopping.")
                break

            with timer("new step"):
                logger.info(f"Beginning step {step_id}")
                status, tiles_df = new_step(algo, basal_step_id, step_id)
                if tiles_df is None:
                    expected_counts[step_id] = 0

            if status in [WorkerStatus.CONVERGED, WorkerStatus.EMPTY_STEP]:
                stopping_indicator += 1
            else:
                stopping_indicator = 0

            if (
                status == WorkerStatus.CONVERGED
                and stopping_indicator >= n_parallel_steps
            ):
                logger.info("Converged. Stopping.")
                break
            elif (
                status == WorkerStatus.EMPTY_STEP
                and stopping_indicator >= n_parallel_steps
            ):
                logger.info(f"{n_parallel_steps} empty step. Stopping.")
                break

            with timer("submit packets"):
                logger.info("Submitting packets for step %d", step_id)
                processing_tasks.put(
                    (step_id, await submit_packet_df(backend, algo, tiles_df))
                )

            with timer("wait for packets"):
                if processing_tasks.qsize() > n_parallel_steps - 1:
                    tasks_step_id, tasks = processing_tasks.get()
                    if tasks_step_id in expected_counts:
                        assert len(tasks) == 0
                    else:
                        expected_counts[tasks_step_id] = await wait_for_packets(
                            backend, algo, tasks
                        )

        with timer("process final packets"):
            while not processing_tasks.empty():
                await wait_for_packets(backend, algo, processing_tasks.get()[1])

        with timer("verify"):
            await asyncio.gather(*[t for t in verify_tasks if t is not None])
            db.verify(step_id)

        assert processing_tasks.empty()
        assert step_id < algo.cfg["n_steps"]
        assert (
            step_id == algo.cfg["n_steps"] - 1 or stopping_indicator >= n_parallel_steps
        )


@contextlib.contextmanager
def timer(name):
    start = time.time()
    logger.debug(f"Starting timing block: {name}")
    yield
    logger.debug(f"{name} took {time.time() - start:.3f} seconds")


class Backend(abc.ABC):
    algo_cfg_entries = [
        "init_K",
        "n_K_double",
        "max_K",
        "tile_batch_size",
        "lam",
        "delta",
        "global_target",
        "max_target",
        "bootstrap_seed",
        "nB",
        "alpha",
        "calibration_min_idx",
        "clickhouse_service",
        "job_name",
    ]

    def entrypoint(self, algo_type, kwargs):
        """
        Passing control of the entrypoint to the backend allows executing the
        leader somewhere besides the launching machine.

        The default behavior is to just run the leader locally.
        """
        return entrypoint(self, algo_type, kwargs)

    @abc.abstractmethod
    def connect_to_db(self, job_name):
        pass

    @abc.abstractmethod
    def get_cfg(self):
        pass

    @abc.abstractmethod
    @contextlib.asynccontextmanager
    async def setup(self, algo):
        pass

    @abc.abstractmethod
    def submit_tiles(self, tiles_df):
        pass

    @abc.abstractmethod
    async def wait_for_results(self, awaitable):
        pass


class LocalBackend(Backend):
    def __init__(self, use_clickhouse: bool = False):
        self.use_clickhouse = use_clickhouse

    def connect_to_db(self, cfg):
        if self.use_clickhouse:
            import confirm.cloud.clickhouse as ch

            if cfg["clickhouse_service"] is None:
                raise ValueError(
                    "clickhouse_service must be set when using Clickhouse."
                )
            return ch.ClickhouseTiles.connect(
                job_name=cfg["job_name"], service=cfg["clickhouse_service"]
            )
        else:
            if cfg["job_name"].startswith("unnamed"):
                db_filepath = ":memory:"
            else:
                db_filepath = cfg["job_name"] + ".db"

            return DuckDBTiles.connect(path=db_filepath)

    def get_cfg(self):
        return {"use_clickhouse": self.use_clickhouse}

    @contextlib.asynccontextmanager
    async def setup(self, algo):
        self.algo = algo
        yield

    def sync_submit_tiles(self, tiles_df, refine_deepen: bool, report: dict):
        return process_tiles(self.algo, tiles_df, refine_deepen, report)

    async def submit_tiles(self, tiles_df, refine_deepen: bool, report: dict):
        return self.sync_submit_tiles(tiles_df, refine_deepen, report)

    async def wait_for_results(self, results):
        return results


def print_report(report, _db):
    ready = report.copy()
    for k in ready:
        if (
            isinstance(ready[k], float)
            or isinstance(ready[k], np.floating)
            or isinstance(ready[k], jax.Array)
        ):
            ready[k] = f"{ready[k]:.6f}"
    logger.debug(pformat(ready))
