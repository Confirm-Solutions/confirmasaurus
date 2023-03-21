import asyncio
import logging
import time
from enum import Enum

import numpy as np
import pandas as pd

import imprint as ip

logger = logging.getLogger(__name__)


async def process_packet_set(backend, algo, packets):
    coros = []
    for step_id, packet_id in packets:
        logger.debug(
            "waiting for work: (step_id=%s, packet_id=%s)",
            step_id,
            packet_id,
        )
        packet_df = algo.db.get_packet(step_id, packet_id)
        if packet_df.shape[0] == 0:
            logger.warning(
                f"Packet is empty or already processed."
                f" (step_id={step_id}, packet_id={packet_id})"
            )
        else:
            coros.append(process_packet(backend, algo, packet_df))
    await asyncio.gather(*coros)


async def process_packet_df(backend, algo, tiles_df):
    if tiles_df is None or tiles_df.shape[0] == 0:
        return
    coros = []
    for packet_id, packet_df in tiles_df.groupby("packet_id"):
        coros.append(process_packet(backend, algo, packet_df=packet_df))
    await asyncio.gather(*coros)


async def process_packet(backend, algo, packet_df):
    start = time.time()
    report = dict()
    report["worker_id"] = algo.cfg["worker_id"]
    report["step_id"] = packet_df.iloc[0]["step_id"]
    report["packet_id"] = packet_df.iloc[0]["packet_id"]
    report["n_tiles"] = packet_df.shape[0]
    report["n_total_sims"] = packet_df["K"].sum()

    start = time.time()
    logger.debug("Processing %d tiles.", packet_df.shape[0])
    results_df, runtime_simulating = await backend.process_tiles(packet_df)

    report["runtime_simulating"] = runtime_simulating
    report["runtime_per_sim_ns"] = (
        report["runtime_simulating"] / report["n_total_sims"] * 1e9
    )
    report["time"] = time.time()
    report["runtime_process_tiles"] = time.time() - start
    algo.db.insert_results(results_df, algo.get_orderer())
    status = WorkerStatus.WORKING
    report["status"] = status.name
    report["runtime_total"] = time.time() - start
    algo.callback(report, algo.db)
    algo.db.insert_report(report)


async def new_step(algo, basal_step_id, new_step_id):
    start = time.time()
    status, tiles_df, report = await _new_step(algo, basal_step_id, new_step_id)
    report["worker_id"] = algo.cfg["worker_id"]
    report["status"] = status.name
    report["basal_step_id"] = basal_step_id
    report["step_id"] = new_step_id
    report["n_packets"] = tiles_df["packet_id"].nunique() if tiles_df is not None else 0
    report["runtime_total"] = time.time() - start
    algo.callback(report, algo.db)
    algo.db.insert_report(report)
    return status, tiles_df


async def _new_step(algo, basal_step_id, new_step_id):
    report = dict()

    converged, convergence_data = await algo.convergence_criterion(
        basal_step_id, report
    )

    start = time.time()
    report["runtime_convergence_criterion"] = time.time() - start
    if converged:
        logger.debug("Convergence!!")
        return WorkerStatus.CONVERGED, None, report
    elif new_step_id >= algo.cfg["n_steps"]:
        logger.debug("Reached maximum number of steps. Terminating.")
        return WorkerStatus.REACHED_N_STEPS, None, report

    n_existing_packets = algo.db.n_existing_packets(new_step_id)
    if n_existing_packets is not None and n_existing_packets > 0:
        logger.debug(
            f"Step {new_step_id} already exists with"
            f" {n_existing_packets} packets. Skipping."
        )
        return (
            WorkerStatus.ALREADY_EXISTS,
            algo.db.get_packet(new_step_id),
            report,
        )

    # If we haven't converged, we create a new step.
    selection_df = await algo.select_tiles(
        basal_step_id, new_step_id, report, convergence_data
    )

    if selection_df is None:
        # New step is empty so we have terminated but
        # failed to converge.
        logger.debug("New step is empty despite failure to converge.")
        return WorkerStatus.EMPTY_STEP, None, report, [], []

    selection_df["finisher_id"] = algo.cfg["worker_id"]
    selection_df["active"] = ~(selection_df["refine"] | selection_df["deepen"])

    # NOTE: this is a pathway towards eventually having variable splitting
    # logic?
    dim = ip.Grid(selection_df, None).d
    selection_df["refine"] = selection_df["refine"].astype(int) * (2**dim)
    selection_df["deepen"] = selection_df["deepen"].astype(int) * 2

    if "split" not in selection_df.columns:
        selection_df["split"] = False
    done_cols = [
        "step_id",
        "packet_id",
        "id",
        "active",
        "finisher_id",
        "refine",
        "deepen",
        "split",
    ]
    done_df = selection_df[done_cols]
    algo.db.insert_done(done_df)

    n_refine = (selection_df["refine"] > 0).sum()
    n_deepen = (selection_df["deepen"] > 0).sum()
    report.update(
        dict(
            n_refine=n_refine,
            n_deepen=n_deepen,
            n_complete=selection_df["active"].sum(),
        )
    )

    nothing_to_do = n_refine == 0 and n_deepen == 0
    if nothing_to_do:
        logger.debug(
            "No tiles are refined or deepened in this step."
            " Marking these parent tiles as finished and trying again."
        )
        return WorkerStatus.NO_NEW_TILES, None, report

    # Actually deepen and refine!
    g_new = refine_and_deepen(
        selection_df, algo.null_hypos, algo.cfg["max_K"], algo.cfg["worker_id"]
    )
    g_new.df["step_id"] = new_step_id
    g_new.df["creator_id"] = algo.cfg["worker_id"]
    g_new.df["creation_time"] = ip.timer.simple_timer()

    # there might be new inactive tiles that resulted from splitting with
    # the null hypotheses. we need to mark these tiles as finished.
    inactive_df = g_new.df[~g_new.df["active"]].copy()
    inactive_df["packet_id"] = np.int32(-1)
    inactive_done = inactive_df.copy()
    inactive_done["refine"] = 0
    inactive_done["deepen"] = 0
    inactive_done["split"] = True
    inactive_done["finisher_id"] = algo.cfg["worker_id"]
    inactive_done = inactive_done[done_cols].copy()
    algo.db.insert_tiles(inactive_df)
    algo.db.insert_done(inactive_done)

    # Assign tiles to packets and then insert them into the database for
    # processing.
    g_active = g_new.prune_inactive()

    def assign_packets(df):
        return pd.Series(
            np.floor(np.arange(df.shape[0]) / algo.cfg["packet_size"]).astype(int),
            df.index,
        )

    g_active.df["packet_id"] = assign_packets(g_active.df)
    algo.db.insert_tiles(g_active.df)
    report["time"] = time.time()

    logger.debug(
        f"Starting step {new_step_id}" f" with {g_active.n_tiles} tiles to simulate."
    )
    return WorkerStatus.NEW_STEP, g_active.df, report


def refine_and_deepen(df, null_hypos, max_K, worker_id):
    g_deepen_in = ip.Grid(df.loc[(df["deepen"] > 0) & (df["K"] < max_K)], worker_id)
    g_deepen = ip.grid._raw_init_grid(
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

    g_refine_in = ip.grid.Grid(df.loc[df["refine"] > 0], worker_id)
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


class WorkerStatus(Enum):
    # Statuses which terminate the worker.
    CONVERGED = 0
    REACHED_N_STEPS = 1
    # Statuses which end the self-help stage.
    EMPTY_STEP = 2
    NO_NEW_TILES = 3
    # Normal solo work statuses.
    NEW_STEP = 4
    WORKING = 5
    ALREADY_EXISTS = 6
    EMPTY_PACKET = 7

    def done(self):
        return (
            (self == WorkerStatus.REACHED_N_STEPS)
            or (self == WorkerStatus.CONVERGED)
            or (self == WorkerStatus.EMPTY_STEP)
        )
