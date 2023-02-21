import asyncio
import logging
import time

import numpy as np
import pandas as pd

import imprint.timer
from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.init import _launch_task

logger = logging.getLogger(__name__)


async def process_packet_set(algo, zone_id, packets):
    coros = [
        process_packet(algo, zone_id, step_id, packet_id)
        for step_id, packet_id in packets
    ]
    tasks = await asyncio.gather(*coros)
    insert_tasks, report_tasks = zip(*tasks)
    await asyncio.gather(*insert_tasks)
    return report_tasks


async def process_packet(algo, zone_id, step_id, packet_id):
    report = dict()
    status, insert_results = await _process(algo, zone_id, step_id, packet_id, report)
    report["status"] = status.name
    algo.callback(report, algo.db)
    insert_report = await _launch_task(algo.db, algo.db.insert_report, report)
    return insert_results, insert_report


async def _process(algo, zone_id, step_id, packet_id, report):
    report["worker_id"] = algo.cfg["worker_id"]
    report["zone_id"] = zone_id
    report["step_id"] = step_id
    report["packet_id"] = packet_id

    logger.debug(
        "waiting for work: (zone_id=%s, step_id=%s, packet_id=%s)",
        zone_id,
        step_id,
        packet_id,
    )
    get_work = await _launch_task(
        algo.db, algo.db.get_packet, zone_id, step_id, packet_id
    )
    work = await get_work

    if work.shape[0] == 0:
        logger.warning(
            f"Packet is empty or already processed. (zone_id={zone_id}, "
            f"step_id={step_id}, packet_id={packet_id})"
        )
        return WorkerStatus.EMPTY_PACKET, asyncio.sleep(0)

    report["n_tiles"] = work.shape[0]

    start = time.time()
    results_df = algo.process_tiles(tiles_df=work, report=report)
    report["runtime_process_tiles"] = time.time() - start

    insert_results = await _launch_task(
        algo.db, algo.db.insert_results, results_df, algo.get_orderer()
    )
    logger.debug(
        "Processed %d tiles in %0.2f seconds.",
        work.shape[0],
        report["runtime_process_tiles"],
    )
    return WorkerStatus.WORKING, insert_results


async def new_step(algo, zone_id, new_step_id):
    status, report, n_packets = await _new_step(algo, zone_id, new_step_id)
    report["status"] = status.name
    report["zone_id"] = zone_id
    report["step_id"] = new_step_id
    insert_report = await _launch_task(algo.db, algo.db.insert_report, report)
    algo.callback(report, algo.db)
    return status, n_packets, insert_report


async def _new_step(algo, zone_id, new_step_id):
    report = dict()

    start = time.time()
    converged, convergence_data = algo.convergence_criterion(zone_id, report)
    report["runtime_convergence_criterion"] = time.time() - start
    if converged:
        logger.debug("Convergence!!")
        return WorkerStatus.CONVERGED, report, dict()
    elif new_step_id >= algo.cfg["n_steps"]:
        logger.debug("Reached maximum number of steps. Terminating.")
        # NOTE: no need to coordinate with other workers. They will reach
        # n_steps on their own time.
        return WorkerStatus.REACHED_N_STEPS, report, dict()

    # If we haven't converged, we create a new step.
    start = time.time()
    logger.debug(f"Selecting tiles for step {new_step_id}.")
    selection_df = algo.select_tiles(zone_id, new_step_id, report, convergence_data)
    report["runtime_select_tiles"] = time.time() - start

    if selection_df is None:
        # New step is empty so we have terminated but
        # failed to converge.
        logger.debug("New step is empty despite failure to converge.")
        return WorkerStatus.EMPTY_STEP, report, dict()

    selection_df["finisher_id"] = algo.cfg["worker_id"]
    selection_df["active"] = ~(selection_df["refine"] | selection_df["deepen"])
    if "split" not in selection_df.columns:
        selection_df["split"] = False
    done_cols = [
        "zone_id",
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

    n_refine = selection_df["refine"].sum()
    n_deepen = selection_df["deepen"].sum()
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
        return WorkerStatus.NO_NEW_TILES, report, dict(n_packets=0, n_tiles=0)

    # Actually deepen and refine!
    g_new = refine_and_deepen(
        selection_df, algo.null_hypos, algo.cfg["max_K"], algo.cfg["worker_id"]
    )
    g_new.df["zone_id"] = np.uint32(zone_id)
    g_new.df["step_id"] = new_step_id
    g_new.df["creator_id"] = algo.cfg["worker_id"]
    g_new.df["creation_time"] = imprint.timer.simple_timer()

    # there might be new inactive tiles that resulted from splitting with
    # the null hypotheses. we need to mark these tiles as finished.
    def insert_inactive():
        inactive_df = g_new.df[~g_new.df["active"]].copy()
        inactive_df["packet_id"] = np.int32(-1)
        algo.db.insert_tiles(inactive_df)
        inactive_df["refine"] = False
        inactive_df["deepen"] = False
        inactive_df["split"] = True
        inactive_df["finisher_id"] = algo.cfg["worker_id"]
        algo.db.finish(inactive_df[done_cols])

    # Assign tiles to packets and then insert them into the database for
    # processing.
    g_active = g_new.prune_inactive()

    def assign_packets(df):
        return pd.Series(
            np.floor(np.arange(df.shape[0]) / algo.cfg["packet_size"]).astype(int),
            df.index,
        )

    g_active.df["packet_id"] = assign_packets(g_active.df)
    n_packets = g_active.df["packet_id"].max() + 1

    await asyncio.gather(
        await _launch_task(algo.db, algo.db.finish, done_df),
        await _launch_task(algo.db, insert_inactive),
        await _launch_task(algo.db, algo.db.insert_tiles, g_active.df),
    )
    logger.debug(
        f"For zone {zone_id}, starting step {new_step_id}"
        f" with {g_active.n_tiles} tiles to simulate."
    )
    return WorkerStatus.NEW_STEP, report, n_packets


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
    inherit_cols = ["K", "coordination_id"]
    # TODO: it's possible to do better by refining by more than just a
    # factor of 2.
    g_refine = g_refine_in.refine(inherit_cols)

    # NOTE: Instead of prune_alternative here, we mark alternative tiles as
    # inactive. This means that we will have a full history of grid
    # construction.
    out = g_refine.concat(g_deepen).add_null_hypos(null_hypos, inherit_cols)
    out.df.loc[out._which_alternative(), "active"] = False
    return out


def verify_adagrid(db):
    _ = db.get_tiles()
    results_df = db.get_results()

    duplicate_ids = results_df["id"].value_counts()
    assert duplicate_ids.max() == 1

    inactive_ids = results_df.loc[~results_df["active"], "id"]
    assert inactive_ids.unique().shape == inactive_ids.shape

    parents = results_df["parent_id"].unique()
    parents_that_dont_exist = np.setdiff1d(parents, inactive_ids)
    inactive_tiles_with_no_children = np.setdiff1d(inactive_ids, parents)
    assert parents_that_dont_exist.shape[0] == 1
    assert parents_that_dont_exist[0] == 0
    assert inactive_tiles_with_no_children.shape[0] == 0