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
        kwargs["db"] = DuckDBTiles.connect(kwargs["job_name"])
    db = kwargs["db"]

    try:
        async with contextlib.AsyncExitStack() as stack:
            with timer("init"):
                stack.enter_context(DatabaseLogging(db))
                algo, incomplete_packets, next_step = await init(algo_type, 1, kwargs)

            last_backup = asyncio.Event()
            backup_task = asyncio.create_task(backup(last_backup, db, algo.cfg))

            with timer("backend.setup"):
                await stack.enter_async_context(backend.setup(algo))

            with timer("process_initial_incompletes"):
                await process_packet_set(backend, algo, np.array(incomplete_packets))

            for step_id in range(next_step, algo.cfg["n_steps"]):
                with timer("verify"):
                    await db.verify()

                if time.time() - entry_time > kwargs["timeout"]:
                    logger.info("Job timeout reached, stopping.")
                    break

                with timer("new step"):
                    logger.info(f"Beginning step {step_id}")
                    status, tiles_df = await new_step(algo, step_id, step_id)

                if status == WorkerStatus.CONVERGED:
                    logger.info("Converged. Stopping.")
                    break
                elif status == WorkerStatus.EMPTY_STEP:
                    logger.info("Empty step. Stopping.")
                    break

                with timer("process packets"):
                    await process_packet_df(backend, algo, tiles_df)
    finally:
        if "backup_task" in locals():
            last_backup.set()
            await backup_task

        with timer("verify"):
            await db.verify()
        return db


async def backup(last_backup, db, cfg):
    if cfg["job_name"] is False:
        return
    if cfg["job_name"] is None and not cfg["prod"]:
        return

    done = False
    while not done:
        await asyncio.sleep(0.1)
        if last_backup.is_set():
            logger.info("Performing final backup.")
            done = True
        with timer("backup"):
            try:
                import confirm.cloud.clickhouse as ch

                await ch.backup(cfg["job_name"], db)
            except Exception:
                # If a backup fails for some reason, we don't want that
                # to kill the job.
                logger.exception("Error backing up database", exc_info=True)


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
        try:
            asyncio.get_running_loop()

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
        except RuntimeError:
            pass

        # If we're not running through Jupyter, then we can just run the
        # async entrypoint directly.
        return asyncio.run(async_entrypoint(self, algo_type, kwargs))

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
