import abc
import asyncio
import contextlib
import logging
import time
from pprint import pformat

import jax
import numpy as np

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
    if backend.already_run:
        raise RuntimeError("An adagrid backend can only be used once.")
    backend.already_run = True
    kwargs.update(backend.input_cfg)

    # If we're running through Jupyter, then an event loop will already be
    # running and we're not allowed to start a new event loop inside of the
    # existing one. So we need to run the async entrypoint in a separate
    # thread. synchronicity is a library that makes this easy.
    import synchronicity

    synchronizer = synchronicity.Synchronizer()
    sync_entry = synchronizer.create(async_entrypoint)[synchronicity.Interface.BLOCKING]
    return sync_entry(backend, algo_type, kwargs)


async def async_entrypoint(backend, algo_type, kwargs):
    if kwargs.get("db", None) is None:
        kwargs["db"] = DuckDBTiles.connect()
    with DatabaseLogging(db=kwargs["db"]):
        start = time.time()
        algo, incomplete_packets, zone_steps = await init(
            algo_type, True, 1, kwargs["n_zones"], kwargs
        )
        logger.debug("init took %s", time.time() - start)

        incomplete_packets = np.array(incomplete_packets)
        min_step_completed = min(zone_steps.values())
        max_step_completed = max(zone_steps.values())
        every = algo.cfg["coordinate_every"]
        initial_coordinations = []
        n_zones = algo.cfg["n_zones"]

        def get_next_coord(step_id):
            next_every = step_id + every - (step_id % every)
            if len(initial_coordinations) == 0:
                return next_every
            next_initial = min([i for i in initial_coordinations if i > step_id])
            return min(next_initial, next_every)

        next_coord = get_next_coord(min_step_completed)
        assert next_coord == get_next_coord(max_step_completed)
        start_step = min_step_completed + 1

        start = time.time()
        async with backend.setup(algo_type, algo, kwargs):
            logger.debug("setup took %s", time.time() - start)

            start = time.time()
            await backend.process_initial_incompletes(incomplete_packets)
            logger.debug("process_initial_incompletes took %s", time.time() - start)

            while start_step < algo.cfg["n_steps"]:
                if next_coord == start_step:
                    if n_zones > 1:
                        # If there's more than one zone, then we need to coordinate.
                        # NOTE: coordinations happen *before* the same-named step.
                        # e.g. a coordination at step 5 happens before new_step and
                        # process_packets for 5.
                        start = time.time()
                        coord_status, lazy_tasks, zone_steps = await coordinate(
                            algo, next_coord, n_zones
                        )
                        backend.lazy_tasks.extend(lazy_tasks)
                        logger.debug("coordinate took %s", time.time() - start)
                        if coord_status.done():
                            break

                    next_coord = get_next_coord(next_coord)
                assert next_coord > start_step

                start = time.time()
                end_step = min(next_coord, algo.cfg["n_steps"] + 1)
                statuses = await backend.run_zones(zone_steps, start_step, end_step)
                start_step = end_step
                assert len(statuses) == len(zone_steps)
                logger.debug("run_zones took %s", time.time() - start)

                # If there's only one zone and that zone is done, then we're
                # totally done.
                if len(statuses) == 1:
                    if statuses[0].done():
                        break

        start = time.time()
        # TODO: currently we only verify at the end, should we do it more often?
        verify_task = asyncio.create_task(algo.db.verify())

        await asyncio.gather(*backend.lazy_tasks)

        await verify_task
        logger.debug("verify and lazy_tasks took %s", time.time() - start)
        return algo.db


class Backend(abc.ABC):
    def __init__(self):
        self.already_run = False
        # some variables from input_cfg should not be read directly, instead
        # access algo.cfg['...'] instead. the reason is so that we correctly
        # handle loading a configuration. for example, coordinate_every should
        # not change over the lifetime of a job.
        self.input_cfg = {}

    @abc.abstractmethod
    @contextlib.asynccontextmanager
    async def setup(self, algo_type, algo, kwargs):
        pass

    @abc.abstractmethod
    async def process_initial_incompletes(self, incomplete_packets):
        pass

    @abc.abstractmethod
    async def run_zones(self, zone_steps, start_step, end_step):
        pass


class LocalBackend(Backend):
    def __init__(self, n_zones=1, coordinate_every=5):
        """
        Args:
            n_zones: _description_. Defaults to 1.
            coordinate_every: The number of steps between each distributed coordination.
                This is ignored when n_zones == 1. Defaults to 5.
        """
        super().__init__()
        self.lazy_tasks = []
        self.input_cfg = {"coordinate_every": coordinate_every, "n_zones": n_zones}

    @contextlib.asynccontextmanager
    async def setup(self, algo_type, algo, kwargs):
        self.algo = algo
        yield

    async def process_initial_incompletes(self, incomplete_packets):
        self.lazy_tasks.extend(await process_packet_set(self.algo, incomplete_packets))

    async def run_zones(self, zone_steps, start_step, end_step):
        return await asyncio.gather(
            *[
                self._run_zone(self.algo, zone_id, start_step, end_step)
                for zone_id in zone_steps
            ]
        )

    async def _run_zone(self, algo, zone_id, start_step, end_step):
        if start_step >= end_step:
            return
        start = time.time()
        logger.debug(
            f"Zone {zone_id} running from step {start_step} "
            f"through step {end_step - 1}."
        )
        for step_id in range(start_step, end_step):
            logger.debug(f"Zone {zone_id} beginning step {step_id}")
            status, tiles_df, before_next_step_tasks, lazy_tasks = await new_step(
                algo, zone_id, step_id
            )
            self.lazy_tasks.extend(lazy_tasks)
            self.lazy_tasks.extend(await process_packet_df(algo, tiles_df))
            await asyncio.gather(*before_next_step_tasks)
            if status.done():
                logger.debug(f"Zone {zone_id} finished with status {status}.")
                break
        logger.debug(
            f"_run_zone({zone_id}, {start_step}, {end_step}) "
            f"took {time.time() - start:.2f}"
        )
        return status


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
