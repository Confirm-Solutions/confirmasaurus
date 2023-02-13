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
import codecs
import copy
import json
import platform
import subprocess
import time
import warnings
from pprint import pformat

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import imprint
from .convergence import WorkerStatus
from .db import DuckDBTiles

logger = imprint.log.getLogger(__name__)


class LocalBackend:
    n_workers: int = 1

    def run(self, ada):
        return ada._run_local()


class ModalBackend:
    def __init__(
        self, n_workers=1, gpu="A100", job_name_prefix="adagrid", modal_kwargs=None
    ):
        self.n_workers = n_workers
        self.gpu = gpu
        self.job_name_prefix = job_name_prefix
        self.modal_kwargs = dict() if modal_kwargs is None else modal_kwargs

    def run(self, ada):
        import modal
        import confirm.cloud.clickhouse as ch
        import confirm.cloud.modal_util as modal_util

        model_type = ada.model_type
        algo_type = ada.algo_type
        callback = ada.callback
        job_id = ada.db.job_id

        stub = modal.Stub(f"{self.job_name_prefix}_{job_id}")

        p = modal_util.get_defaults()
        p.update(dict(gpu=self.gpu, serialized=True))

        @stub.function(**p)
        def _modal_adagrid_worker(_):
            modal_util.setup_env()
            db = ch.Clickhouse.connect(job_id=job_id)
            runner = Adagrid(model_type, None, db, algo_type, callback, dict(), dict())
            runner._run_local()

        logger.info(f"Launching Modal job with {self.n_workers} workers")
        with stub.run(show_progress=False):
            _ = list(_modal_adagrid_worker.map(range(self.n_workers)))
        return ada.db


def run(algo_type, locals_):
    ada = Adagrid()
    ada.init(
        locals_["model_type"],
        locals_["g"],
        locals_["db"],
        algo_type,
        locals_["callback"],
        locals_["overrides"],
        locals_["backend"].n_workers,
        locals_,
    )
    return locals_["backend"].run(ada)


class Adagrid:
    """
    Generic entrypoint for running adagrid regardless of whether it is
    calibration or validation. The arguments here will be self-explanatory if
    you understand ada_validate or ada_calibrate.
    """

    async def init(
        self,
        model_type,
        g,
        db,
        algo_type,
        callback,
        overrides,
        n_workers,
        locals_,
        worker_id=None,
    ):
        """
        __init__ is not allowed to be aysnc. This is a workaround. Since we're
        only calling init from two places, it's not a big deal.
        """
        start = time.time()
        self.first_step = True
        self.callback = callback
        self.model_type = model_type
        self.algo_type = algo_type

        ########################################
        # STEP 1: Prepare the grid and database
        ########################################
        if g is None and db is None:
            raise ValueError("Must provide either an initial grid or a database!")

        if db is None:
            db = DuckDBTiles.connect()
        self.db = db

        if worker_id is None:
            worker_id = db.new_worker()
        self.worker_id = worker_id
        imprint.log.worker_id.set(self.worker_id)
        print("11", time.time() - start)

        ########################################
        # STEP 2: Prepare the configuration
        ########################################

        # locals() is a handy way to pass all the arguments from ada_validate, but
        # it has the downside of being very magic.
        if g is None:
            # If we are resuming a job, we need to load the config from the database.
            load_cfg_df = db.store.get("config")
            cfg = load_cfg_df.iloc[0].to_dict()

            # Let's be extra sure to not share worker_id between workers!
            cfg["worker_id"] = None

            model_kwargs = json.loads(cfg["model_kwargs_json"])

            # IMPORTANT: Except for overrides, parameters will be ignored!
            for k in overrides:
                # Some parameters cannot be overridden because the job just wouldn't
                # make sense anymore.
                if k in [
                    "model_seed",
                    "model_kwargs",
                    "alpha",
                    "init_K",
                    "n_K_double",
                    "bootstrap_seed",
                    "nB",
                    "model_name",
                ]:
                    raise ValueError(f"Parameter {k} cannot be overridden.")
                cfg[k] = overrides[k]

        else:
            # Using locals() is a simple way to get all the config vars in the
            # function definition. But, we need to erase fields that are not part
            # of the "config".
            cfg = {
                k: v
                for k, v in locals_.items()
                if k
                not in [
                    "model_type",
                    "g",
                    "db",
                    "overrides",
                    "callback",
                    "model_kwargs",
                    "transformation",
                    "backend",
                ]
            }

            # Storing the model_type is infeasible because it's a class, so we store
            # the model type name instead.
            cfg["model_name"] = model_type.__name__

            model_kwargs = locals_["model_kwargs"]
            if model_kwargs is None:
                model_kwargs = {}
            cfg["model_kwargs_json"] = json.dumps(model_kwargs)

            if overrides is not None:
                warnings.warn("Overrides are ignored when starting a new job.")
        print("12", time.time() - start)

        # Very important not to share worker_id between workers so we overwrite!
        cfg["worker_id"] = self.worker_id

        cfg["jax_platform"] = jax.lib.xla_bridge.get_backend().platform
        default_tile_batch_size = dict(gpu=64, cpu=4)
        cfg["tile_batch_size"] = cfg["tile_batch_size"] or (
            default_tile_batch_size[cfg["jax_platform"]]
        )

        if cfg["packet_size"] is None:
            cfg["packet_size"] = cfg["step_size"]

        ########################################
        # STEP 3: Collect a bunch of system information for later debugging and
        # reproducibility.
        ########################################
        print("13", time.time() - start)
        cfg.update(
            dict(
                git_hash=_run(["git", "rev-parse", "HEAD"]),
                git_diff=_run(["git", "diff", "HEAD"]),
                platform=platform.platform(),
                nvidia_smi=_run(["nvidia-smi"]),
            )
        )
        if cfg["prod"]:
            cfg["pip_freeze"] = _run(["pip", "freeze"])
            cfg["conda_list"] = _run(["conda", "list"])
        else:
            cfg["pip_freeze"] = "skipped because prod=False"
            cfg["conda_list"] = "skipped because prod=False"
        print("14", time.time() - start)

        cfg["max_K"] = cfg["init_K"] * 2 ** cfg["n_K_double"]

        self.cfg = cfg
        cfg_df = pd.DataFrame([cfg])
        print("14.5", time.time() - start)

        from threading import Thread

        helper_loop = asyncio.new_event_loop()

        def f(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = Thread(target=f, args=(helper_loop,))
        t.start()

        # we wrap set_or_append in a lambda so that the db.store access can be
        # run in a separate thread in case the Store __init__ method does any
        # substantial work (e.g. in ClickhouseStore, we create a table)
        wait_for = [
            asyncio.create_task(
                asyncio.to_thread(lambda: db.store.set_or_append("config", cfg_df))
            )
        ]
        print("15", time.time() - start)

        ########################################
        # STEP 5: If we are starting a new job, we need to fill the database
        # with the initial grid.
        ########################################
        if g is not None:
            # Copy the input grid so that the caller is not surprised by any changes.
            df = copy.deepcopy(g.df)
            df["K"] = cfg["init_K"]
            df["step_id"] = 0
            df["creator_id"] = self.worker_id
            df["creation_time"] = imprint.timer.simple_timer()
            print("16", time.time() - start)

            initial_worker_ids = [self.worker_id] + self.db.new_workers(n_workers - 1)
            print("17", time.time() - start)
            # TODO: configurable assignment seed
            assignment_seed = 0
            df["worker_id"] = random_assignments(
                assignment_seed, g.n_tiles, n_workers, initial_worker_ids
            )

            def assign_packets(df):
                return pd.Series(
                    np.floor(np.arange(df.shape[0]) / self.cfg["packet_size"]).astype(
                        int
                    ),
                    df.index,
                )

            df["packet_id"] = df.groupby("worker_id")["worker_id"].transform(
                assign_packets
            )
            print("18", time.time() - start)

            wait_for.append(asyncio.create_task(db.init_tiles(df)))
            # sleeping immediately means that the init_tiles task will be
            # scheduled to run now.
            self.null_hypos = g.null_hypos
            wait_for.append(
                asyncio.create_task(
                    asyncio.to_thread(_store_null_hypos, db, self.null_hypos)
                )
            )

            n_tiles = df.shape[0]
            logger.debug(
                "Initialized database with %d tiles and %d null hypos."
                " The tiles are split between %d workers with packet_size=%s.",
                n_tiles,
                len(self.null_hypos),
                n_workers,
                cfg["packet_size"],
            )
            print("19", time.time() - start)
        else:
            self.null_hypos = _load_null_hypos(self.db)

        ########################################
        # STEP 4: Prepare the model and algorithm
        # These steps need to happen regardless of whether we are starting a new
        # job or resuming an old one.
        # We do this *after* initializing the database so that the two can happen
        # in parallel.
        ########################################

        self.model = model_type(
            seed=cfg["model_seed"],
            max_K=cfg["init_K"] * 2 ** cfg["n_K_double"],
            **model_kwargs,
        )
        self.algo = algo_type(db, self.model, self.null_hypos, self.cfg)
        print("20", time.time() - start)

        await asyncio.gather(*wait_for)

    async def _run_local(self):
        insert_reports = []
        for step_id in range(self.cfg["n_steps"]):
            await self.simulate_step(step_id)
            new_step_id = step_id + 1
            status, report = await self.new_step(new_step_id)
            insert_reports.append(
                asyncio.create_task(asyncio.to_thread(self.db.insert_report, report))
            )
            # If we reached the maximum number of steps, we are completely
            # done.
            if status == WorkerStatus.REACHED_N_STEPS:
                break
            # If we converged or took an empty step, we need to coordinate with
            # the other workers.
            if (status == WorkerStatus.CONVERGED) or (
                status == WorkerStatus.EMPTY_STEP
            ):
                await self.coordinate()

        await asyncio.gather(*insert_reports)

    async def coordinate(self):
        # Goals of coordinate:
        # - update the active and eligible flags.
        # - assign every active and eligible tile to a worker.
        # - need to wait until the DB has actually processed these updates
        # before going back to simulating.
        pass

    async def simulate_step(self, step_id):
        next_work = None
        insert_threads = []
        packet_id = 0
        while True:
            report = dict()
            report["worker_id"] = self.worker_id
            report["step_id"] = step_id
            report["packet_id"] = packet_id

            ########################################
            # Get work
            ########################################
            start = time.time()
            # On the first loop, we won't have queued any work queries yet.
            if next_work is None:
                next_work = asyncio.create_task(
                    asyncio.to_thread(
                        self.db.get_work, self.worker_id, step_id, packet_id
                    )
                )
            work = await next_work
            report["n_tiles"] = work.shape[0]

            # Empty packet is an indication that we are done with this step.
            if work.shape[0] == 0:
                logger.debug("No more work for this step.")
                report["runtime_get_work"] = time.time() - start
                report["status"] = WorkerStatus.WORK_DONE.name
                insert_threads.append(
                    asyncio.create_task(
                        asyncio.to_thread(self.db.insert_report, report)
                    )
                )
                await asyncio.gather(*insert_threads)
                return

            # Queue a query for the next packet.
            next_work = asyncio.create_task(
                asyncio.to_thread(self.db.get_work, self.worker_id, step_id, packet_id)
            )

            # Check if some other worker has already inserted this packet.
            packet_flag = f"{self.db.job_id}:worker_{self.worker_id}_step_{step_id}_packet_{packet_id}_insert"  # noqa
            flag = self.db.redis_con.get(packet_flag)
            if flag is not None:
                logger.debug(
                    "Skipping packet. Flag "
                    f"{packet_flag} is set by worker_id={flag.decode('ascii')}."
                )
                packet_id += 1
                report["runtime_skip_packet"] = time.time() - start
                report["status"] = WorkerStatus.SKIPPED.name
                insert_threads.append(
                    asyncio.create_task(
                        asyncio.to_thread(self.db.insert_report, report)
                    )
                )
                continue
            report["runtime_get_work"] = time.time() - start

            ########################################
            # Process tiles
            ########################################
            start = time.time()
            results_df = self.algo.process_tiles(tiles_df=work, report=report)
            report["runtime_process_tiles"] = time.time() - start
            logger.debug(
                "Simulated %d tiles in %0.2f seconds.",
                work.shape[0],
                report["runtime_process_tiles"],
            )

            ########################################
            # Insert results in a separate thread to avoid blocking the main thread.
            ########################################
            def insert_results(packet_id, packet_flag, report, results_df):
                was_flag_set = self.db.redis_con.setnx(packet_flag, self.worker_id)
                if was_flag_set == 0:
                    logger.warning(
                        f"(step_id={step_id}, packet_id={packet_id})"
                        " already inserted, discarding results."
                    )
                    report["status"] = WorkerStatus.DISCARDED.name
                else:
                    self.db.insert_results(
                        results_df, "total_cost_order, tie_bound_order"
                    )
                    logger.debug(
                        "inserted packet results for "
                        f"(step_id = {step_id}, packet_id={packet_id})"
                        f" with {results_df.shape[0]} results"
                    )
                    report["status"] = WorkerStatus.WORKING.name
                self.db.insert_report(report)

            start = time.time()
            insert_threads.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        insert_results, packet_id, packet_flag, report, results_df
                    )
                )
            )
            report["runtime_insert_results"] = time.time() - start
            packet_id += 1

    async def new_step(self, new_step_id, report):
        report = dict()
        start = time.time()
        converged, convergence_data = self.algo.convergence_criterion(report)
        report["runtime_convergence_criterion"] = time.time() - start

        if converged:
            logger.debug("Convergence!!")
            return WorkerStatus.CONVERGED, report
        elif new_step_id >= self.cfg["n_steps"]:
            logger.debug("Reached maximum number of steps. Terminating.")
            # NOTE: no need to coordinate with other workers. They will reach
            # n_steps on their own time.
            return WorkerStatus.REACHED_N_STEPS, report

        # If we haven't converged, we create a new step.
        start = time.time()
        tiles_df = self.algo.select_tiles(report, convergence_data)
        report["runtime_select_tiles"] = time.time() - start

        if tiles_df is None:
            # New step is empty so we have terminated but
            # failed to converge.
            logger.debug(
                "New step is empty. Waiting for the next "
                "coordination despite failure to converge."
            )
            return WorkerStatus.EMPTY_STEP, report

        tiles_df["finisher_id"] = self.worker_id
        tiles_df["active"] = ~(tiles_df["refine"] | tiles_df["deepen"])
        if "split" not in tiles_df.columns:
            tiles_df["split"] = False
        done_cols = [
            "id",
            "step_id",
            "packet_id",
            "active",
            "finisher_id",
            "refine",
            "deepen",
            "split",
        ]

        done_task = asyncio.create_task(
            asyncio.to_thread(self.db.finish, tiles_df[done_cols])
        )

        n_refine = tiles_df["refine"].sum()
        n_deepen = tiles_df["deepen"].sum()
        report.update(
            dict(
                n_refine=n_refine,
                n_deepen=n_deepen,
                n_complete=tiles_df["active"].sum(),
            )
        )

        nothing_to_do = n_refine == 0 and n_deepen == 0
        if nothing_to_do:
            logger.debug(
                "No tiles are refined or deepened in this step."
                " Marking these parent tiles as finished and trying again."
            )
            return WorkerStatus.NO_NEW_TILES, report

        # Actually deepen and refine!
        g = refine_and_deepen(
            tiles_df, self.null_hypos, self.cfg["max_K"], self.cfg["worker_id"]
        )
        g.df["step_id"] = new_step_id
        g.df["creator_id"] = self.cfg["worker_id"]
        g.df["creation_time"] = imprint.timer.simple_timer()

        # there might be new inactive tiles that resulted from splitting with
        # the null hypotheses. we need to mark these tiles as finished.
        def insert_inactive(inactive_df):
            inactive_df["packet_id"] = np.int32(-1)
            self.db.insert_tiles(inactive_df)
            inactive_df["refine"] = False
            inactive_df["deepen"] = False
            inactive_df["split"] = True
            inactive_df["finisher_id"] = self.cfg["worker_id"]
            self.db.finish(inactive_df[done_cols])

        inactive_df = g.df[~g.df["active"]].copy()
        inactive_task = asyncio.create_task(
            asyncio.to_thread(insert_inactive, inactive_df)
        )

        # Assign tiles to packets and then insert them into the database for
        # processing.
        g_active = g.prune_inactive()

        def assign_packets(df):
            return pd.Series(
                np.floor(np.arange(df.shape[0]) / self.cfg["packet_size"]).astype(int),
                df.index,
            )

        g_active.df["packet_id"] = assign_packets(g_active.df)
        g_active.df["worker_id"] = self.worker_id
        insert_task = asyncio.create_task(
            asyncio.to_thread(self.db.insert_tiles, g_active.df)
        )
        logger.debug(
            f"Starting step {new_step_id} with {g_active.n_tiles} tiles to simulate."
        )
        await asyncio.gather(done_task, inactive_task, insert_task)
        return WorkerStatus.NEW_STEP, report


def random_assignments(seed, n_tiles, n_workers, names):
    # Randomly assign tiles to packets.
    np.random.seed(seed)
    splits = np.array_split(np.arange(n_tiles), n_workers)
    assignment = np.empty(n_tiles, dtype=np.int32)
    for i in range(n_workers):
        assignment[splits[i]] = names[i]
    rng = np.random.default_rng()
    rng.shuffle(assignment)
    return assignment


def refine_and_deepen(df, null_hypos, max_K, worker_id):
    g_deepen_in = imprint.grid.Grid(df.loc[df["deepen"] & (df["K"] < max_K)], worker_id)
    g_deepen = imprint.grid._raw_init_grid(
        g_deepen_in.get_theta(),
        g_deepen_in.get_radii(),
        worker_id=worker_id,
        parents=g_deepen_in.df["id"],
    )

    # We just multiply K by 2 to deepen.
    # TODO: it's possible to do better by multiplying by 4 or 8
    # sometimes when a tile clearly needs *way* more sims. how to
    # determine this?
    g_deepen.df["K"] = g_deepen_in.df["K"] * 2

    g_refine_in = imprint.grid.Grid(df.loc[df["refine"]], worker_id)
    inherit_cols = ["K"]
    # TODO: it's possible to do better by refining by more than just a
    # factor of 2.
    g_refine = g_refine_in.refine(inherit_cols)

    # NOTE: Instead of prune_alternative here, we mark alternative tiles as
    # inactive. This means that we will have a full history of grid
    # construction.
    out = g_refine.concat(g_deepen).add_null_hypos(null_hypos, inherit_cols)
    out.df.loc[out._which_alternative(), "active"] = False
    return out


def _store_null_hypos(db, null_hypos):
    # we need to convert the pickled object to a valid string so that it can be
    # inserted into a database. converting to a from base64 achieves this goal:
    # https://stackoverflow.com/a/30469744/3817027
    serialized = [
        codecs.encode(cloudpickle.dumps(h), "base64").decode() for h in null_hypos
    ]
    desc = [h.description() for h in null_hypos]
    df = pd.DataFrame({"serialized": serialized, "description": desc})
    db.store.set("null_hypos", df)


def _load_null_hypos(db):
    df = db.store.get("null_hypos")
    null_hypos = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        null_hypos.append(
            cloudpickle.loads(codecs.decode(row["serialized"].encode(), "base64"))
        )
    return null_hypos


def verify_adagrid(df):
    duplicate_ids = df["id"].value_counts()
    assert duplicate_ids.max() == 1

    inactive_ids = df.loc[~df["active"], "id"]
    assert inactive_ids.unique().shape == inactive_ids.shape

    parents = df["parent_id"].unique()
    parents_that_dont_exist = np.setdiff1d(parents, inactive_ids)
    inactive_tiles_with_no_children = np.setdiff1d(inactive_ids, parents)
    assert parents_that_dont_exist.shape[0] == 1
    assert parents_that_dont_exist[0] == 0
    assert inactive_tiles_with_no_children.shape[0] == 0


def _run(cmd):
    try:
        return (
            subprocess.check_output(" ".join(cmd), stderr=subprocess.STDOUT, shell=True)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.returncode} {exc.output}"


def print_report(_iter, report, _db):
    ready = report.copy()
    for k in ready:
        if (
            isinstance(ready[k], float)
            or isinstance(ready[k], np.floating)
            or isinstance(ready[k], jnp.DeviceArray)
        ):
            ready[k] = f"{ready[k]:.6f}"
    logger.debug(pformat(ready))
