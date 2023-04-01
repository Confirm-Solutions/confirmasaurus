import abc
import contextlib
import logging
import queue
import time
from pprint import pformat

import jax
import numpy as np
import synchronicity

from .asyncio_runner import Runner
from .db import DatabaseLogging
from .db import DuckDBTiles
from .init import init
from .step import new_step
from .step import process_packet_set
from .step import submit_packet_df
from .step import wait_for_packets
from .step import WorkerStatus

logger = logging.getLogger(__name__)

synchronizer = synchronicity.Synchronizer()


def pass_control_to_backend(algo_type, kwargs):
    backend = kwargs.get("backend", None)
    if backend is None:
        backend = LocalBackend()
    kwargs.update(backend.get_cfg())

    return backend.entrypoint(algo_type, kwargs)


def setup_db(cfg, require_fresh=False):
    if cfg["job_name"] is None:
        if cfg["job_name_prefix"] is None:
            cfg["job_name_prefix"] = "unnamed"
        cfg["job_name"] = f"{cfg['job_name_prefix']}_{time.strftime('%Y%m%d_%H%M%S')}"

    print(list(cfg.keys()))
    ch_client = None
    if cfg["clickhouse_service"] is not None:
        import confirm.cloud.clickhouse as ch

        ch_client = ch.connect(
            job_name=cfg["job_name"], service=cfg["clickhouse_service"]
        )
        if require_fresh and ch.does_table_exist(ch_client, "tiles"):
            raise ValueError(
                "ClickHouse table 'tiles' already exists. "
                "Please choose a fresh job_name."
            )

    if cfg["job_name"].startswith("unnamed"):
        db_filepath = ":memory:"
    else:
        db_filepath = cfg["job_name"] + ".db"

    db = DuckDBTiles.connect(path=db_filepath, ch_client=ch_client)
    if require_fresh and db.does_table_exist("tiles"):
        raise ValueError(
            "DuckDB table 'tiles' already exists." " Please choose a fresh job_name."
        )
    return db


def entrypoint(backend, algo_type, kwargs):
    """
    Passing control of the entrypoint to the backend allows executing the
    leader somewhere besides the launching machine.

    The default behavior is to just run the leader locally.
    """

    db = setup_db(kwargs, require_fresh=True)
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
                runner.run(db.ch_wait())
                assert len(db.ch_tasks) == 0
    return db


async def async_entrypoint(backend, db, algo_type, kwargs):
    entry_time = time.time()
    with timer("init"):
        algo, incomplete_packets, next_step = init(db, algo_type, kwargs)

    start = time.time()
    async with backend.setup(algo):
        logger.debug(f"backend.setup() took {time.time() - start:.2f} seconds")

        with timer("process_initial_incompletes"):
            await process_packet_set(backend, algo, np.array(incomplete_packets))

        stopping_indicator = 0
        n_parallel_steps = algo.cfg["n_parallel_steps"]
        processing_tasks = queue.Queue()
        for step_id in range(next_step, algo.cfg["n_steps"]):
            basal_step_id = max(step_id - n_parallel_steps, 0)

            with timer("verify"):
                db.verify(basal_step_id)

            if time.time() - entry_time > kwargs["timeout"]:
                logger.info("Job timeout reached, stopping.")
                break

            with timer("new step"):
                logger.info(f"Beginning step {step_id}")
                status, tiles_df = new_step(algo, basal_step_id, step_id)

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

            with timer("process packets"):
                logger.info("Processing packets for step %d", step_id)
                processing_tasks.put(await submit_packet_df(backend, algo, tiles_df))
                if processing_tasks.qsize() > n_parallel_steps - 1:
                    await wait_for_packets(backend, algo, processing_tasks.get())

            with timer("Check on Clickhouse inserts"):
                await db.ch_checkup()

        with timer("process final packets"):
            while not processing_tasks.empty():
                await wait_for_packets(backend, algo, processing_tasks.get())

        with timer("verify"):
            db.verify(basal_step_id)

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
    def get_cfg(self):
        return {}

    @contextlib.asynccontextmanager
    async def setup(self, algo):
        self.algo = algo
        yield

    def sim_tiles(self, tiles_df):
        report = dict()
        report["sim_start_time"] = time.time()
        tbs = self.algo.cfg["tile_batch_size"]
        if tbs is None:
            tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
        report["tile_batch_size"] = tbs
        start = time.time()
        results_df = self.algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)
        report["runtime_simulating"] = time.time() - start
        report["sim_done_time"] = time.time()
        return results_df, report

    async def submit_tiles(self, tiles_df):
        self.algo.db.ch_insert("tiles", tiles_df, create=False)
        results_df, report = self.sim_tiles(tiles_df)
        self.algo.db.ch_insert("results", results_df, create=True)
        # NOTE: We restrict the set of columns returned to the leader. Why?
        # 1) The full results data has already been written to Clickhouse.
        # 2) The leader only needs a subset of the results in order to
        #    determine future adagrid steps.
        # 3) The leader can fill in a lot of the tile details by joining with
        #    the tiles table.
        return_cols = ["id"] + self.algo.get_important_columns()
        return results_df[return_cols], report

    async def wait_for_results(self, awaitable):
        return awaitable


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
