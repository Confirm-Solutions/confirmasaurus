import copy
import json
import logging
import platform
import subprocess
import time
import warnings
from pprint import pformat

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import imprint
from .convergence import WorkerStatus
from .db import DuckDBTiles

logger = logging.getLogger(__name__)


def run(modeltype, g, db, locals_, algo_type, callback):
    if g is None and db is None:
        raise ValueError("Must provide either an initial grid or a database!")

    if db is None:
        db = DuckDBTiles.connect()

    worker_id = db.new_worker()
    imprint.log.worker_id.set(worker_id)

    # locals() is a handy way to pass all the arguments to prepare_config, but
    # it has the downside of being too magic.
    g = locals_["g"]
    overrides = locals_["overrides"]

    if g is None:
        load_cfg_df = db.store.get("config")
        cfg = load_cfg_df.iloc[0].to_dict()
        # Very important not to share worker_id between workers!!
        cfg["worker_id"] = None
        model_kwargs = json.loads(cfg["model_kwargs_json"])
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
                "modeltype",
                "g",
                "db",
                "overrides",
                "callback",
                "model_kwargs",
                "transformation",
            ]
        }
        cfg["model_name"] = locals_["modeltype"].__name__

        model_kwargs = locals_["model_kwargs"]
        if model_kwargs is None:
            model_kwargs = {}
        cfg["model_kwargs_json"] = json.dumps(model_kwargs)

        if overrides is not None:
            warnings.warn("Overrides are ignored when starting a new job.")

    cfg["worker_id"] = worker_id
    cfg["jax_backend"] = jax.lib.xla_bridge.get_backend().platform
    cfg["tile_batch_size"] = cfg["tile_batch_size"] or (
        64 if cfg["jax_backend"] == "gpu" else 4
    )

    if cfg["packet_size"] is None:
        cfg["packet_size"] = cfg["step_size"]

    cfg.update(
        dict(
            git_hash=_run(["git", "rev-parse", "HEAD"]),
            git_diff=_run(["git", "diff", "HEAD"]),
            platform=platform.platform(),
            nvidia_smi=_run(["nvidia-smi"]),
        )
    )
    if locals_["prod"]:
        cfg["pip_freeze"] = _run(["pip", "freeze"])
        cfg["conda_list"] = _run(["conda", "list"])
    else:
        cfg["pip_freeze"] = "skipped because prod=False"
        cfg["conda_list"] = "skipped because prod=False"

    cfg_df = pd.DataFrame([cfg])
    db.store.set_or_append("config", cfg_df)

    model = modeltype(
        seed=cfg["model_seed"],
        max_K=cfg["init_K"] * 2 ** cfg["n_K_double"],
        **model_kwargs,
    )
    null_hypos = _load_null_hypos(db) if g is None else g.null_hypos

    algo = algo_type(db, model, null_hypos, cfg)

    if g is not None:
        # Copy the input grid so that the caller is not surprised by any changes.
        df = copy.deepcopy(g.df)
        df["K"] = cfg["init_K"]
        df["step_id"] = 0
        df["step_iter"], n_packets = step_iter_assignments(df, cfg["packet_size"])
        df["creator_id"] = worker_id
        df["creation_time"] = imprint.timer.simple_timer()

        db.init_tiles(df)
        _store_null_hypos(db, null_hypos)

        n_tiles = df.shape[0]
        logger.debug(
            f"first step {(0, 0, n_packets, n_tiles)} "
            f"n_tiles={n_tiles} packet_size={cfg['packet_size']}"
        )
        db.set_step_info(step_id=0, step_iter=0, n_iter=n_packets, n_tiles=n_tiles)

    if cfg["n_iter"] == 0:
        return 0, [], db

    reports = []
    for worker_iter in range(cfg["n_iter"]):
        try:
            report = dict(worker_iter=worker_iter, worker_id=worker_id)
            start = time.time()
            status, work_df, report = run_iter(algo, db, report, cfg["n_steps"])
            if work_df is not None and work_df.shape[0] > 0:
                logger.debug("Processing %s tiles.", work_df.shape[0])
                start = time.time()
                results = algo.process_tiles(tiles_df=work_df, report=report)
                report["runtime_processing"] = time.time() - start
                db.insert_results(results, algo.get_orderer())
            report["status"] = status.name
            report["runtime_full_iter"] = time.time() - start

            if callback is not None:
                callback(worker_iter, report, db)
            reports.append(report)

            if status.done():
                # We just stop this worker. The other workers will continue to run
                # but will stop once they reached the convergence criterion check.
                # It might be faster to stop all the workers immediately, but that
                # would be more engineering effort.
                break
        except KeyboardInterrupt:
            # TODO: we want to die robustly when there's a keyboard interrupt.
            print("KeyboardInterrupt")
            return worker_iter, reports, db

    return worker_iter + 1, reports, db


def run_iter(algo, db, report, n_steps):
    """
    One iteration of the adagrid algorithm.

    Parallelization:
    There are four main portions of the adagrid algo:
    - Convergence criterion --> serialize to avoid data races.
    - Tile selection --> serialize to avoid data races.
    - Deciding whether to refine or deepen --> parallelize.
    - Simulation and CSE --> parallelize.
    By using a distributed lock around the convergence/tile selection, we can have
    equality between workers and not need to have a leader node.

    Args:
        i: The iteration number.

    Returns:
        done: A value from the Enum Convergence:
             (INCOMPLETE, CONVERGED, FAILED).
        report: A dictionary of information about the iteration.
    """

    start = time.time()

    ########################################
    # Step 1: Get the next batch of tiles to process. We do this before
    # checking for convergence because part of the convergence criterion is
    # whether there are any impossible tiles.
    ########################################
    max_loops = 25
    i = 0
    status = WorkerStatus.WORK
    report["runtime_wait_for_lock"] = 0
    report["runtime_wait_for_work"] = 0
    while i < max_loops:
        logger.debug("Starting loop %s.", i)
        report["waitings"] = i
        with db.lock:
            logger.debug("Claimed DB lock.")
            report["runtime_wait_for_lock"] += time.time() - start
            start = time.time()

            step_id, step_iter, step_n_iter, step_n_tiles = db.get_step_info()
            report["step_id"] = step_id
            report["step_iter"] = step_iter
            report["step_n_iter"] = step_n_iter
            report["step_n_tiles"] = step_n_tiles

            # Check if there are iterations left in this step.
            # If there are, get the next batch of tiles to process.
            if step_iter < step_n_iter:
                logger.debug("get_work(step_id=%s, step_iter=%s)", step_id, step_iter)
                work_df = db.get_work(step_id, step_iter)
                report["runtime_get_work"] = time.time() - start
                start = time.time()
                report["work_extraction_time"] = time.time()
                report["n_processed"] = work_df.shape[0]
                logger.debug("get_work(...) returned %s tiles.", work_df.shape[0])

                if work_df.shape[0] > 0:
                    # If there's work, update the step info and return the work!
                    report["runtime_update_step_info"] = time.time() - start
                    db.set_step_info(
                        step_id=step_id,
                        step_iter=step_iter + 1,
                        n_iter=step_n_iter,
                        n_tiles=step_n_tiles,
                    )
                    # Why do we return the work instead of just processing it here?
                    # Because the database is currently locked and we would
                    # like to release the lock while we process tiles!
                    return status, work_df, report
                else:
                    # If step_iter < step_n_iter but there's no work, then
                    # The INSERT into tiles that was supposed to populate
                    # the work is probably incomplete. We should wait a
                    # very short time and try again.
                    logger.debug("No work despite step_iter < step_n_iter.")
                    wait = 0.1

            # If there are no iterations left in the step, we check if the
            # step is complete. For a step to be complete means that all
            # tiles have results.
            else:
                n_processed_tiles = db.n_processed_tiles(step_id)
                report["n_finished_tiles"] = n_processed_tiles
                if n_processed_tiles == step_n_tiles:
                    # If a packet has just been completed, we check for convergence.
                    status, convergence_data = algo.convergence_criterion(report=report)
                    report["runtime_convergence_criterion"] = time.time() - start
                    start = time.time()
                    if status:
                        logger.debug("Convergence!!")
                        return WorkerStatus.CONVERGED, None, report

                    if step_id >= n_steps - 1:
                        # We've completed all the steps, so we're done.
                        logger.debug("Reached max number of steps. Terminating.")
                        return WorkerStatus.REACHED_N_STEPS, None, report

                    # If we haven't converged, we create a new step.
                    new_step_id = algo.new_step(step_id + 1, report, convergence_data)

                    report["runtime_new_step"] = time.time() - start
                    start = time.time()
                    if new_step_id == "empty":
                        # New packet is empty so we have terminated but
                        # failed to converge.
                        logger.debug(
                            "New packet is empty. Terminating despite "
                            "failure to converge."
                        )
                        return WorkerStatus.FAILED, None, report
                    else:
                        # Successful new packet. We should check for work again
                        # immediately.
                        status = WorkerStatus.NEW_STEP
                        wait = 0
                        logger.debug("Successfully created new packet.")
                else:
                    # No work available, but the packet is incomplete. This is
                    # because other workers have claimed all the work but have not
                    # processsed yet.
                    # In this situation, we should release the lock and wait for
                    # other workers to finish.
                    wait = 1
                    logger.debug(
                        "No work available, but packet is incomplete"
                        " with %s/%s tiles complete.",
                        n_processed_tiles,
                        step_n_tiles,
                    )
        if wait > 0:
            logger.debug("Waiting %s seconds and checking for work again.", wait)
            time.sleep(wait)
        if i > 3:
            logger.warning(
                "Worker s has been waiting for work for"
                " %s iterations. This might indicate a bug.",
                i,
            )
        report["runtime_wait_for_work"] += time.time() - start
        i += 1

    return WorkerStatus.STUCK, None, report


def step_iter_assignments(df, packet_size):
    # Randomly assign tiles to packets.
    # TODO: this could be improved to better balance the load.
    # There are two load balancing concerns:
    # - a packet with tiles with a given K value should have lots of
    #   that K so that we have enough work to saturate GPU threads.
    # - we should divide the work evenly among the workers.
    n_tiles = df.shape[0]
    n_packets = int(np.ceil(n_tiles / packet_size))
    splits = np.array_split(np.arange(n_tiles), n_packets)
    assignment = np.empty(n_tiles, dtype=np.int32)
    for i in range(n_packets):
        assignment[splits[i]] = i
    rng = np.random.default_rng()
    rng.shuffle(assignment)
    return assignment, n_packets


def refine_and_deepen(g, null_hypos, max_K, worker_id):
    g_deepen_in = imprint.grid.Grid(g.loc[g["deepen"] & (g["K"] < max_K)], worker_id)
    g_deepen = imprint.grid.init_grid(
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

    g_refine_in = imprint.grid.Grid(g.loc[g["refine"]], worker_id)
    inherit_cols = ["K"]
    # TODO: it's possible to do better by refining by more than just a
    # factor of 2.
    g_refine = g_refine_in.refine(inherit_cols)

    ########################################
    # Step 7: Simulate the new tiles and write to the DB.
    ########################################
    return g_refine.concat(g_deepen).add_null_hypos(null_hypos, inherit_cols).prune()


# TODO: move towards NullHypo class
def _load_null_hypos(db):
    d = db.dimension()
    null_hypos_df = db.store.get("null_hypos")
    null_hypos = []
    for i in range(null_hypos_df.shape[0]):
        n = np.array([null_hypos_df[f"n{i}"].iloc[i] for i in range(d)])
        c = null_hypos_df["c"].iloc[i]
        null_hypos.append(imprint.grid.HyperPlane(n, c))
    return null_hypos


def _store_null_hypos(db, null_hypos):
    d = db.dimension()
    n_hypos = len(null_hypos)
    cols = {f"n{i}": [null_hypos[j].n[i] for j in range(n_hypos)] for i in range(d)}
    cols["c"] = [null_hypos[j].c for j in range(n_hypos)]
    null_hypos_df = pd.DataFrame(cols)
    db.store.set("null_hypos", null_hypos_df)


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