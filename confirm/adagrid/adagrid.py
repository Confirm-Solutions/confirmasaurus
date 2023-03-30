import abc
import asyncio
import contextlib
import logging
import queue
import time
import uuid
from pprint import pformat

import jax
import numpy as np
import synchronicity

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


def entrypoint(algo_type, kwargs):
    backend = kwargs.get("backend", None)
    if backend is None:
        backend = LocalBackend()
    kwargs.update(backend.get_cfg())
    return backend.entrypoint(algo_type, kwargs)


def setup_db(cfg):
    print(list(cfg.keys()))
    ch_client = None
    if cfg["clickhouse_service"] is not None:
        import confirm.cloud.clickhouse as ch

        ch_client = ch.connect(
            job_name=cfg["job_name"], service=cfg["clickhouse_service"]
        )

    db = cfg.get("db", None)
    if db is None:
        if cfg["job_name"] is None:
            db_filepath = ":memory:"
            cfg["job_name"] = "unnamed_" + uuid.uuid4().hex
        else:
            db_filepath = f"{cfg['job_name']}.db"
        db = DuckDBTiles.connect(path=db_filepath, ch_client=ch_client)
    db.ch_client = ch_client
    return db


# synchronizer is used so that we can run our event loop code in a separate
# thread regardless of whether an event loop is already running.
# If we're not running through Jupyter, then no event loop is running and we
# can just run the async entrypoint directly:
#   asyncio.run(async_entrypoint(self, algo_type, kwargs))
#
# But, if we're running through Jupyter, then an event loop will already be
# running and we're not allowed to start a new event loop inside of the
# existing one. So we need to run the async entrypoint in a separate
# thread. synchronicity is a library that makes this easy.
# @synchronizer.create_blocking


async def async_entrypoint(backend, algo_type, kwargs):
    entry_time = time.time()

    db = setup_db(kwargs)
    kwargs["db"] = db

    async with contextlib.AsyncExitStack() as stack:
        with timer("init"):
            stack.enter_context(DatabaseLogging(db))
            algo, incomplete_packets, next_step = init(algo_type, kwargs)

        with timer("backend.setup"):
            await stack.enter_async_context(backend.setup(algo))

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
                processing_tasks.put(submit_packet_df(backend, algo, tiles_df))
                await asyncio.sleep(0)
                if processing_tasks.qsize() > n_parallel_steps - 1:
                    await wait_for_packets(backend, algo, processing_tasks.get())

            with timer("wait for backup"):
                await db.ch_wait()

    with timer("verify"):
        db.verify(basal_step_id)
    return db


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
        return asyncio.run(async_entrypoint(self, algo_type, kwargs))

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

    def submit_tiles(self, tiles_df):
        report = dict()
        report["sim_start_time"] = time.time()
        self.algo.db.ch_insert("tiles", tiles_df, create=False)
        tbs = self.algo.cfg["tile_batch_size"]
        if tbs is None:
            tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
        report["tile_batch_size"] = tbs
        start = time.time()
        results_df = self.algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)
        self.algo.db.ch_insert("results", results_df, create=True)
        report["runtime_simulating"] = time.time() - start
        report["sim_done_time"] = time.time()
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
