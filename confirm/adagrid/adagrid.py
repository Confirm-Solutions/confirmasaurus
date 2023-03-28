import abc
import asyncio
import contextlib
import logging
import queue
import time
from pprint import pformat

import jax
import numpy as np

from .db import DatabaseLogging
from .db import DuckDBTiles
from .init import init
from .step import new_step
from .step import process_packet_set
from .step import submit_packet_df
from .step import wait_for_packets
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

    async with contextlib.AsyncExitStack() as stack:
        with timer("init"):
            stack.enter_context(DatabaseLogging(db))
            algo, incomplete_packets, next_step = init(algo_type, kwargs)

        with timer("backend.setup"):
            await stack.enter_async_context(backend.setup(algo))

        check_backup = await stack.enter_async_context(backup_daemon(db, algo.cfg))

        with timer("process_initial_incompletes"):
            await process_packet_set(backend, algo, np.array(incomplete_packets))

        stopping_indicator = 0
        n_parallel_steps = algo.cfg["n_parallel_steps"]
        processing_tasks = queue.Queue()
        for step_id in range(next_step, algo.cfg["n_steps"]):
            check_backup()

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

    with timer("verify"):
        db.verify(basal_step_id)
    return db


@contextlib.asynccontextmanager
async def backup_daemon(db, cfg):
    if (cfg["job_name"] is False) or (cfg["job_name"] is None and not cfg["prod"]):
        yield lambda: None
        return

    import confirm.cloud.clickhouse as ch

    last_backup = asyncio.Event()

    async def _daemon():
        ch_client = await asyncio.to_thread(ch.connect, cfg["job_name"])
        done = False
        first_time = {k: True for k in ch.all_tables}
        while not done:
            await asyncio.sleep(0.1)
            if last_backup.is_set():
                logger.info("Performing final backup.")
                done = True
            with timer("backup"):
                try:
                    await ch.backup(ch_client, db, first_time)
                except Exception:
                    # If a backup fails for some reason, we don't want that
                    # to kill the job.
                    logger.exception("Error backing up database", exc_info=True)

    backup_task = asyncio.create_task(_daemon())

    def check_backup():
        try:
            # by checking for a result, we can cause exception to be raised
            # now instead of waiting for the whole job to be done.
            backup_task.result()
        except asyncio.InvalidStateError:
            # if the backup_task is still running, that's great and we can
            # ignore this!
            pass

    yield check_backup

    last_backup.set()
    await backup_task


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
            run_directly = False
        except RuntimeError:
            run_directly = True

        if run_directly:
            # If we're not running through Jupyter, then we can just run the
            # async entrypoint directly.
            return asyncio.run(async_entrypoint(self, algo_type, kwargs))
        else:
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
        tbs = self.algo.cfg["tile_batch_size"]
        if tbs is None:
            tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
        logger.debug("Processing tiles using tile batch size %s", tbs)
        start = time.time()
        out = self.algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)
        return out, time.time() - start

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
