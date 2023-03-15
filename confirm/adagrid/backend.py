import abc
import asyncio
import contextlib
import logging
import time
from pprint import pformat

import jax
import numpy as np

from .convergence import WorkerStatus
from .coordinate import coordinate
from .db import DatabaseLogging
from .db import DuckDBTiles
from .init import init
from .step import new_step
from .step import process_packet_df
from .step import process_packet_set

logger = logging.getLogger(__name__)


def entrypoint(algo_type, kwargs):
    backend = kwargs.get("backend", None)
    if backend is None:
        backend = LocalBackend()
    kwargs.update(backend.get_cfg())

    # If we're running through Jupyter, then an event loop will already be
    # running and we're not allowed to start a new event loop inside of the
    # existing one. So we need to run the async entrypoint in a separate
    # thread. synchronicity is a library that makes this easy.
    import synchronicity

    synchronizer = synchronicity.Synchronizer()
    sync_entry = synchronizer.create(async_entrypoint)[synchronicity.Interface.BLOCKING]
    return sync_entry(backend, algo_type, kwargs)


async def async_entrypoint(backend, algo_type, kwargs):
    entry_time = time.time()
    if kwargs.get("db", None) is None:
        kwargs["db"] = DuckDBTiles.connect()
    db = kwargs["db"]
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        all_lazy_tasks = await stack.enter_async_context(lazy_handler())
        stack.enter_context(DatabaseLogging(db))

        with timer("init"):
            algo, incomplete_packets, zone_steps = await init(
                algo_type, True, 1, kwargs["n_zones"], kwargs
            )
            every = algo.cfg["coordinate_every"]
            n_zones = algo.cfg["n_zones"]

        await stack.enter_async_context(
            backup_daemon(db, algo.cfg["prod"], algo.cfg["job_name"])
        )

        with timer("setup"):
            await stack.enter_async_context(backend.setup(algo_type, algo, kwargs))

        def get_next_coord(step_id):
            return step_id + every - (step_id % every)

        min_step_completed = min(zone_steps.values())
        max_step_completed = max(zone_steps.values())
        next_coord = get_next_coord(min_step_completed)
        assert next_coord == get_next_coord(max_step_completed)
        start_step = min_step_completed + 1

        with timer("process_initial_incompletes"):
            await process_packet_set(backend, algo, np.array(incomplete_packets))

        while start_step < algo.cfg["n_steps"]:
            if time.time() - entry_time > kwargs["timeout"]:
                logger.info("Job timeout reached, stopping.")
                pass

            if n_zones > 1:
                if next_coord == start_step:
                    # If there's more than one zone, then we need to coordinate.
                    # NOTE: coordinations happen *before* the same-named step.
                    # e.g. a coordination at step 5 happens before new_step and
                    # process_packets for 5.
                    with timer("coordinate"):
                        coord_status, lazy_tasks, zone_steps = await coordinate(
                            algo, next_coord, n_zones
                        )
                        all_lazy_tasks.extend(lazy_tasks)

                    if coord_status.done():
                        break

                    next_coord = get_next_coord(next_coord)
                    assert next_coord > start_step
                end_step = min(next_coord, algo.cfg["n_steps"] + 1)
            else:
                end_step = algo.cfg["n_steps"] + 1

            with timer("run_zones"):
                out = await asyncio.gather(
                    *[
                        run_zone(backend, algo, zone_id, start_step, end_step)
                        for zone_id in zone_steps
                    ]
                )
                statuses, lazy_tasks = zip(*out)
                all_lazy_tasks.extend(sum(lazy_tasks, []))
                start_step = end_step
                assert len(statuses) == len(zone_steps)

            # If there's only one zone and that zone is done, then we're
            # totally done.
            if len(statuses) == 1:
                if statuses[0].done():
                    break

        with timer("verify and lazy_tasks"):
            # TODO: currently we only verify at the end, should we do it more often?
            verify_task = asyncio.create_task(algo.db.verify())
            await verify_task

        return algo.db


async def run_zone(backend, algo, zone_id, start_step, end_step):
    if start_step >= end_step:
        return WorkerStatus.REACHED_N_STEPS, []
    all_lazy_tasks = []

    with timer("run_zone(%s, %s, %s)" % (zone_id, start_step, end_step)):
        logger.debug(
            f"Zone {zone_id} running from step {start_step} "
            f"through step {end_step - 1}."
        )
        for step_id in range(start_step, end_step):
            logger.debug(f"Zone {zone_id} beginning step {step_id}")
            status, tiles_df, before_next_step_tasks, lazy_tasks = await new_step(
                algo, zone_id, step_id
            )
            all_lazy_tasks.extend(lazy_tasks)
            all_lazy_tasks.extend(await process_packet_df(backend, algo, tiles_df))
            await asyncio.gather(*before_next_step_tasks)
            if status.done():
                logger.debug(f"Zone {zone_id} finished with status {status}.")
                break
    return status, all_lazy_tasks


@contextlib.asynccontextmanager
async def backup_daemon(db, prod: bool, job_name: str, backup_interval: int = 10 * 60):
    def backup():
        logger.info("Backing up database")
        if job_name is False:
            return
        if job_name is None and not prod:
            return
        import confirm.cloud.clickhouse as ch

        ch_db = ch.Clickhouse.connect(job_name)
        ch.backup(db, ch_db)
        logger.info("Backup complete")

    async def backup_repeater():
        # We want to backup once no matter what. This is so that we always
        # backup a job that finishes in less time than `backup_interval`.
        while True:
            await asyncio.sleep(backup_interval)
            backup()

    task = asyncio.create_task(backup_repeater())
    yield
    # It's okay to cancel here because the backup task is in the same
    # thread as this code and therefore we will never interrupt a backup by
    # canceling.
    task.cancel()
    # We always want to backup before exiting.
    backup()


@contextlib.contextmanager
def timer(name):
    start = time.time()
    yield
    logger.debug(f"{name} took {time.time() - start:.3f} seconds")


@contextlib.asynccontextmanager
async def lazy_handler():
    all_lazy_tasks = []
    yield all_lazy_tasks
    for task in all_lazy_tasks:
        assert isinstance(task, asyncio.Task)
    await asyncio.gather(*all_lazy_tasks)


class Backend(abc.ABC):
    @abc.abstractmethod
    @contextlib.asynccontextmanager
    async def setup(self, algo_type, algo, kwargs):
        pass

    @abc.abstractmethod
    def get_cfg(self):
        pass

    @abc.abstractmethod
    async def process_tiles(self, tiles_df):
        pass


class LocalBackend(Backend):
    @contextlib.asynccontextmanager
    async def setup(self, algo_type, algo, kwargs):
        self.algo = algo
        yield

    def get_cfg(self):
        return {}

    async def process_tiles(self, tiles_df):
        tbs = self.algo.cfg["tile_batch_size"]
        if tbs is None:
            tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
        return await self.algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)


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
