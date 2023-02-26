"""
# Running non-adagrid jobs using the adagrid code.

It is possible to run calibration or validation *without adagrid* using the
interface provided by ada_validate and ada_calibrate! In order to do these,
follow the examples given in test_calibration_nonadagrid_using_adagrid and
test_validation_nonadagrid_using_adagrid. This is useful for several reasons:
- we can use the database backend.
- we can run distributed non-adagrid jobs. 
  
# Distributed adagrid and database backend

## Principles and goals:
- Simplicity.
- Append-only and no updates. This makes the "story" of an adagrid run very
  clear after the fact which is useful both for replicability and debugging.
- Results are written to DB ASAP. This avoids data loss.
- Minimize coordination overhead, maximize performance!
- No leader node. I don't want to have to be responsible for keeping the launch
  node running.

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
import asyncio
import logging
from pprint import pformat

import jax.numpy as jnp
import numpy as np

from confirm.adagrid.coordinate import coordinate
from confirm.adagrid.init import init
from confirm.adagrid.step import new_step
from confirm.adagrid.step import process_packet_set

logger = logging.getLogger(__name__)


class LocalBackend:
    def __init__(self, n_zones=1, coordinate_every=5):
        """
        _summary_

        Args:
            n_zones: _description_. Defaults to 1.
            coordinate_every: The number of steps between each distributed coordination.
                This is ignored for local runs. Defaults to 5.
        """
        self.lazy_tasks = []
        # self.input_cfg should not be read directly, instead access
        # algo.cfg['...'] instead. the reason is so that we correctly handle
        # loading a configuration. for example, coordinate_every should not
        # change over the lifetime of a job.
        # TODO: this feels sketchy...
        # TODO: n_zones would be changeable if we fixed the selection rule and
        # convergence criterion during each coordination.
        self.input_cfg = {"coordinate_every": coordinate_every, "n_zones": n_zones}

    def run(self, algo_type, kwargs):
        kwargs.update(self.input_cfg)
        return maybe_start_event_loop(self._run_async(algo_type, kwargs))

    async def _run_async(self, algo_type, kwargs):
        # it's okay to use input_cfg['n_zones'] here because it's only used at
        # the start of a job.
        algo, incomplete_packets, zone_steps = await init(
            algo_type, 1, self.input_cfg["n_zones"], kwargs
        )
        n_zones = algo.cfg["n_zones"]
        incomplete_packets = np.array(incomplete_packets)
        self.lazy_tasks.extend(await process_packet_set(algo, incomplete_packets))

        min_step_completed = min(zone_steps.values())
        max_step_completed = max(zone_steps.values())
        print("min_step_completed", min_step_completed)

        # if next_coord is 5, do we coordinate after or before step 5?
        # --> before!
        # min_step_completed = 4, max_step_completed = 4, next_coord = 5
        # min_step_completed = 1, max_step_completed = 3, next_coord = 5
        every = algo.cfg["coordinate_every"]
        next_coord = get_next_coord(min_step_completed, every)
        assert next_coord == get_next_coord(max_step_completed, every)
        first_step = min_step_completed + 1

        while first_step < algo.cfg["n_steps"]:
            last_step = min(next_coord, algo.cfg["n_steps"])
            statuses = await asyncio.gather(
                *[
                    self._run_zone(algo, zone_id, first_step, last_step + 1)
                    for zone_id in zone_steps
                ]
            )

            # If there's only one zone and that zone is done, then we're
            # totally done.
            if len(statuses) == 1:
                if statuses[0].done():
                    break

            if n_zones > 1:
                # If there's more than one zone, then we need to coordinate.
                # NOTE: coordinations happen *before* the same-named step.
                # e.g. a coordination at step 5 happens before new_step and
                # process_packets for 5.
                coord_status, lazy_tasks, zone_steps = await coordinate(
                    algo, last_step + 1, n_zones
                )
                self.lazy_tasks.extend(lazy_tasks)
                if coord_status.done():
                    break

            first_step = last_step + 1
            next_coord += every

        await asyncio.gather(*self.lazy_tasks)
        self.lazy_tasks = []
        return algo.db

    async def _run_zone(self, algo, zone_id, start_step_id, end_step_id):
        if start_step_id >= end_step_id:
            return
        logger.debug(
            f"Zone {zone_id} running from step {start_step_id} "
            f"through step {end_step_id - 1}."
        )
        for step_id in range(start_step_id, end_step_id):
            status, n_packets, report_task = await new_step(algo, zone_id, step_id)
            self.lazy_tasks.append(report_task)
            self.lazy_tasks.extend(
                await process_packet_set(
                    algo, [(zone_id, step_id, i) for i in range(n_packets)]
                )
            )
            if status.done():
                logger.debug(f"Zone {zone_id} finished with status {status}.")
                break
        return status


def get_next_coord(step_id, every):
    return step_id + every - (step_id % every)


def maybe_start_event_loop(coro):
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


class ModalBackend:
    def __init__(self, n_workers=1):
        self.n_workers = n_workers

    #     async def run(self, ada):
    #         import confirm.cloud.clickhouse as ch

    #         model_type = ada.model_type
    #         algo_type = ada.algo_type
    #         callback = ada.callbackk
    #         job_id = ada.db.job_id

    #         def _worker():
    #             db = ch.Clickhouse.connect(job_id=job_id)
    #             runner = Adagrid(model_type, None, db, algo_type, callback,
    #                dict(), dict())
    #             runner._run_local()

    # async def run(self, ada, initial_worker_ids):
    #     import modal
    #     import confirm.cloud.clickhouse as ch
    #     import confirm.cloud.modal_util as modal_util

    #     assert len(initial_worker_ids) == self.n_workers

    #     job_id = ada.db.job_id
    #     algo_type = ada.algo_type
    #     pass_params = dict(
    #         callback=ada.callback,
    #         model_type=ada.model_type,
    #         overrides=dict(),
    #         backend=LocalBackend(),
    #         g=None,
    #     )

    #     stub = modal.Stub(f"{self.job_name_prefix}_{job_id}")

    #     p = modal_util.get_defaults()
    #     p.update(dict(gpu=self.gpu, serialized=True))

    #     @stub.function(**p)
    #     def _modal_adagrid_worker(i):
    #         modal_util.setup_env()
    #         kwargs = copy.deepcopy(pass_params)
    #         kwargs["db"] = ch.Clickhouse.connect(job_id=job_id)
    #         kwargs["worker_id"] = initial_worker_ids[i]
    #         asyncio.run(init_and_run(algo_type, **kwargs))

    #     logger.info(f"Launching Modal job with {self.n_workers} workers")
    #     with stub.run(show_progress=False):
    #         _ = list(_modal_adagrid_worker.map(range(self.n_workers)))
    #     return ada.db


def print_report(report, _db):
    ready = report.copy()
    for k in ready:
        if (
            isinstance(ready[k], float)
            or isinstance(ready[k], np.floating)
            or isinstance(ready[k], jnp.DeviceArray)
        ):
            ready[k] = f"{ready[k]:.6f}"
    logger.debug(pformat(ready))
