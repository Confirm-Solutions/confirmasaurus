import abc
import asyncio
import contextlib
import logging
import time
from pprint import pformat

import jax
import numpy as np

from .db import DatabaseLogging
from .db import DuckDBTiles
from .init import init
from .step import new_step
from .step import process_packet_df
from .step import process_packet_set
from .step import WorkerStatus

logger = logging.getLogger(__name__)


def entrypoint(algo_type, kwargs):
    backend = kwargs.get("backend", None)
    if backend is None:
        backend = LocalBackend()
    kwargs.update(backend.get_cfg())
    return backend.entrypoint(algo_type, kwargs)


async def async_entrypoint(backend, algo_type, kwargs):
    entry_time = time.time()

    if kwargs.get("db", None) is None:
        if kwargs["job_name"] is None:
            db_filepath = ":memory:"
        else:
            db_filepath = f"{kwargs['job_name']}.db"
        kwargs["db"] = DuckDBTiles.connect(db_filepath)
    db = kwargs["db"]

    async with contextlib.AsyncExitStack() as stack:
        with timer("init"):
            stack.enter_context(DatabaseLogging(db))
            algo, incomplete_packets, next_step = await init(algo_type, 1, kwargs)

        with timer("backend.setup"):
            await stack.enter_async_context(backend.setup(algo))

        with timer("process_initial_incompletes"):
            await process_packet_set(backend, algo, np.array(incomplete_packets))

        for step_id in range(next_step, algo.cfg["n_steps"]):
            with timer("verify"):
                verify_task = asyncio.create_task(db.verify())
                await verify_task

            if time.time() - entry_time > kwargs["timeout"]:
                logger.info("Job timeout reached, stopping.")
                pass

            with timer("new step"):
                logger.debug(f"Beginning step {step_id}")
                basal_step_id = get_basal_step(step_id, algo.cfg["n_parallel_steps"])
                status, tiles_df = await new_step(algo, basal_step_id, step_id)

            with timer("process packets"):
                await process_packet_df(backend, algo, tiles_df)

            with timer("backup"):
                if (
                    algo.cfg["backup_interval"] is not None
                    and algo.cfg["job_name"] is not False
                    and (algo.cfg["job_name"] is not None or algo.cfg["prod"])
                    and step_id % algo.cfg["backup_interval"] == 0
                ):
                    import confirm.cloud.clickhouse as ch

                    ch.backup(algo.cfg["job_name"], db)

            if status == WorkerStatus.REACHED_N_STEPS:
                logger.debug(f"Reached n_steps={algo.cfg['n_steps']}. Stopping.")
                break
            elif status == WorkerStatus.CONVERGED and step_id == basal_step_id + 1:
                logger.debug("Converged. Stopping.")
                break
            elif status == WorkerStatus.EMPTY_STEP and step_id == basal_step_id + 1:
                logger.debug("Empty step. Stopping.")
                break

        with timer("verify"):
            verify_task = asyncio.create_task(db.verify())
            await verify_task

        return db


def get_basal_step(step_id, n_parallel_steps):
    """
    >>> id = np.arange(1, 10)
    >>> basal = np.maximum(3 * (np.arange(1, 10) // 3) - 1, 0)
    >>> np.stack((id, basal), axis=-1)
    array([[1, 0],
           [2, 0],
           [3, 2],
           [4, 2],
           [5, 2],
           [6, 5],
           [7, 5],
           [8, 5],
           [9, 8]])
    """
    return max(n_parallel_steps * (step_id // n_parallel_steps) - 1, 0)


@contextlib.contextmanager
def timer(name):
    start = time.time()
    yield
    logger.debug(f"{name} took {time.time() - start:.3f} seconds")


class Backend(abc.ABC):
    algo_cfg_entries = [
        "init_K",
        "n_K_double",
        "tile_batch_size",
        "lam",
        "delta",
        "worker_id",
        "global_target",
        "max_target",
        "bootstrap_seed",
        "nB",
        "alpha",
        "calibration_min_idx",
    ]

    def entrypoint(self, algo_type, kwargs):
        """
        Passing control of the entrypoint to the backend allows executing the
        leader somewhere besides the launching machine.

        The default behavior is to just run the leader locally.
        """
        # If we're running through Jupyter, then an event loop will already be
        # running and we're not allowed to start a new event loop inside of the
        # existing one. So we need to run the async entrypoint in a separate
        # thread. synchronicity is a library that makes this easy.
        import synchronicity

        synchronizer = synchronicity.Synchronizer()
        sync_entry = synchronizer.create(async_entrypoint)[
            synchronicity.Interface.BLOCKING
        ]
        return sync_entry(self, algo_type, kwargs)

    @abc.abstractmethod
    def get_cfg(self):
        pass

    @abc.abstractmethod
    @contextlib.asynccontextmanager
    async def setup(self, algo):
        pass

    @abc.abstractmethod
    async def process_tiles(self, tiles_df):
        pass


class LocalBackend(Backend):
    def get_cfg(self):
        return {}

    @contextlib.asynccontextmanager
    async def setup(self, algo):
        self.algo = algo
        yield

    async def process_tiles(self, tiles_df):
        tbs = self.algo.cfg["tile_batch_size"]
        if tbs is None:
            tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
        logger.debug("Processing tiles using tile batch size %s", tbs)
        start = time.time()
        out = await self.algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)
        return out, time.time() - start


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
