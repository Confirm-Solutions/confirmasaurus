"""
# Running non-adagrid jobs using the adagrid code.

It is possible to run calibration or validation *without adagrid* using the
interface provided by ada_validate and ada_calibrate! In order to do these,
follow the examples given in test_calibration_nonadagrid_using_adagrid and
test_validation_nonadagrid_using_adagrid. This is useful for several reasons:
- we can use the database backend.
- we can run distributed non-adagrid jobs. 
  
# Distributed adagrid and database backend

## Design Principles and goals:
- Simplicity.
- Append-only and no updates to the extent possible. This makes the "story" of
  an adagrid run very clear after the fact which is useful both for
  replicability and debugging.
- Results are written to DB ASAP. This avoids data loss.
- Minimize communication overhead, maximize performance! Keep communication
  isolated to the coordination stage.
- Assume that the DB does not lose inserts
- Good to separate computation from communications.
- Idempotency is the most important property of each stage in the pipeline. -->
  we should be able to kill any stage at any point and then re-run it and get
  the same answer.
- Define properties that the system should have at each stage and verify those
  properties as much as possible.

## Databases for Adagrid

Adagrid depends heavily on a database backend.

### Database tables
- tiles: The tiles table contains all the tiles that have been created. These
  tiles may not necessarily have been simulated yet.
- results: Simulation results plus all columns from `tiles` are duplicated. For
  example, in calibration, this will contain lambda* values for each tiles. We
  duplicate columns in `tiles` so that after a run is complete, the `results`
  table can be used as a single source of information about what happened in a
  run.
- done: After a tile has been refined or deepened, information about the event
  is described in the `done` table. Presence in the `done` table indicates that
  the tile is no longer active.
- config: Columns for arguments to the `ada_validate` and `ada_calibrate`
  functions plus other system configuration information.
- the different database backends use other tables when needed. Planar null
  hypotheses are stored in the database.

### Worker ID

Every worker that joins an adagrid job receives a sequential worker ID. 
- worker_id=0 is reserved and should not be used
- worker_id=1 is the default used for non-adagrid, non-distributed jobs
  (e.g. ip.validate, ip.calibrate). It is also used for the initializing worker
  in distributed adagrid jobs.
- worker_id>=2 are workers that join the adagrid job.

### DuckDB

DuckDB is an embedded database used as the default single process backend for
the adagrid code. It feels a lot like SQLite but is much faster for our use case.

### Clickhouse

Clickhouse is a distributed database that is used as the distributed backend. It's 
a bit slower than DuckDB. We run a Clickhouse server on Clickhouse Cloud. 

### Upstash Redis

Clickhouse does not support locking or transactions. We use Upstash, a
serverless Redis provider, for distributed locks that allow us to avoid data
races and other parallelization problems.

Medis and RedisInsight are Mac GUIs for Redis. Useful for debugging!

## Adagrid algorithm high-level overview.

The adagrid algorithm proceeds in a "steps" which each consist of:
1. Convergence criterion. 
2. Tile selection: which tiles will be used during this step.
3. Tile creation: for each tile, will we refine or deepen? Then, create the new
    tiles. When deepening a tile, we actually create a new tile and
    deactivate the old tile.
4. Simulation: for each tile, simulate the model and then 

The first three of these steps must be done in serial to preserve replicability
and correctness. We effectively put these steps inside a "critical" code region
using a distributed lock. The simulation step is parallelized across all
workers.
"""
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
    coro = async_entrypoint(backend, algo_type, kwargs)

    # Check if an event loop is already running. If we're running through
    # Jupyter, then an event loop will already be running and we're not allowed
    # to start a new event loop inside of the existing one.
    try:
        loop = asyncio.get_running_loop()
        existing_loop = True
    except RuntimeError:
        existing_loop = False

    # We run here instead of under the except statement so that any
    # exceptions are not prefixed with "During handling of the above
    # exception..." messages.
    if existing_loop:
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)


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
        self.algo_type = algo_type
        self.algo = algo
        self.kwargs = kwargs
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
