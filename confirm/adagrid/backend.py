import abc
import asyncio
import contextlib
import logging
import time
from pprint import pformat

import jax
import numpy as np

from .convergence import WorkerStatus
from .db import DatabaseLogging
from .db import DuckDBTiles
from .init import init
from .step import coordinate
from .step import new_step
from .step import process_packet_df
from .step import process_packet_set

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
        kwargs["db"] = DuckDBTiles.connect()
    db = kwargs["db"]
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        stack.enter_context(DatabaseLogging(db))

        with timer("init"):
            algo, incomplete_packets, zone_steps = await init(
                algo_type, 1, kwargs["n_zones"], kwargs
            )
            every = algo.cfg["coordinate_every"]
            n_zones = algo.cfg["n_zones"]

        await stack.enter_async_context(
            backup_daemon(
                db, algo.cfg["prod"], algo.cfg["job_name"], algo.cfg["backup_interval"]
            )
        )

        with timer("setup"):
            await stack.enter_async_context(backend.setup(algo))

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
                        coord_status, zone_steps = await coordinate(
                            algo, next_coord, n_zones
                        )

                    if coord_status.done():
                        break

                    next_coord = get_next_coord(next_coord)
                    assert next_coord > start_step
                end_step = min(next_coord, algo.cfg["n_steps"] + 1)
            else:
                end_step = algo.cfg["n_steps"] + 1

            with timer("run_zones"):
                statuses = await asyncio.gather(
                    *[
                        run_zone(backend, algo, zone_id, start_step, end_step)
                        for zone_id in zone_steps
                    ]
                )
                start_step = end_step
                assert len(statuses) == len(zone_steps)

            with timer("verify"):
                verify_task = asyncio.create_task(algo.db.verify())
                await verify_task

            # If there's only one zone and that zone is done, then we're
            # totally done.
            if len(statuses) == 1:
                if statuses[0].done():
                    break

        with timer("verify"):
            verify_task = asyncio.create_task(algo.db.verify())
            await verify_task

        return algo.db


async def run_zone(backend, algo, zone_id, start_step, end_step):
    if start_step >= end_step:
        return WorkerStatus.REACHED_N_STEPS, []

    with timer("run_zone(%s, %s, %s)" % (zone_id, start_step, end_step)):
        logger.debug(
            f"Zone {zone_id} running from step {start_step} "
            f"through step {end_step - 1}."
        )
        for step_id in range(start_step, end_step):
            logger.debug(f"Zone {zone_id} beginning step {step_id}")
            status, tiles_df = await new_step(algo, zone_id, step_id)
            await process_packet_df(backend, algo, tiles_df)
            if status.done():
                logger.debug(f"Zone {zone_id} finished with status {status}.")
                break
    return status


@contextlib.asynccontextmanager
async def backup_daemon(db, prod: bool, job_name: str, backup_interval: int):
    if backup_interval is None:
        yield
        return

    async def backup():
        if job_name is False:
            return
        if job_name is None and not prod:
            return

        try:
            import confirm.cloud.clickhouse as ch

            logger.info("Backing up database")
            ch_db = ch.connect(job_name)
            await ch.backup(ch_db, db)
            logger.info(f"Backup complete to {ch_db.database}")
        except Exception:
            # If one backup fails for some reason, we don't want that to stop
            # the backup daemon or to kill the job.
            logger.exception("Error backing up database")

    repeat = True

    async def backup_repeater():
        while repeat:
            await asyncio.sleep(backup_interval)
            await backup()

    task = asyncio.create_task(backup_repeater())
    yield
    # It's okay to cancel here because the backup task is in the same
    # thread as this code and therefore we will never interrupt a backup by
    # canceling.
    repeat = False
    task.cancel()
    # We always want to backup before exiting.
    await backup()


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
