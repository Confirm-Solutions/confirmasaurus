import asyncio
import logging
import time
from enum import Enum

import numpy as np
import pandas as pd

import imprint as ip
from .const import MAX_STEP

logger = logging.getLogger(__name__)


async def process_packet_set(backend, algo, packets):
    wait_for = []
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
            wait_for.append(await submit_packet(backend, algo, packet_df))
    await wait_for_packets(backend, algo, wait_for)


async def submit_packet_df(backend, algo, tiles_df):
    if tiles_df is None or tiles_df.shape[0] == 0:
        return []
    packets = []
    for packet_id, packet_df in tiles_df.groupby("packet_id"):
        packets.append(await submit_packet(backend, algo, packet_df))
    return packets


async def wait_for_packets(backend, algo, packets):
    coros = []
    for p in packets:
        coros.append(wait_for_packet(backend, algo, *p))
    logger.debug("Waiting for packets to finish.")
    reports = await asyncio.gather(*coros)
    logger.debug("Packets finished.")
    algo.db.insert_reports(reports)


async def submit_packet(backend, algo, packet_df):
    start = time.time()
    report = dict()
    report["start_time"] = start
    report["step_id"] = packet_df.iloc[0]["step_id"]
    report["packet_id"] = packet_df.iloc[0]["packet_id"]
    report["n_tiles"] = packet_df.shape[0]
    report["n_total_sims"] = packet_df["K"].sum()

    logger.debug("Submitting %d tiles for processing.", packet_df.shape[0])
    # NOTE: we do not restrict the list of columns sent to the worker! This is
    # because the tile insert to Clickhouse happens on the worker.
    # Justification:
    # - we either need to insert tiles into Clickhouse from the leader or the worker.
    # - we must send a minimal amount of data to the worker in order to simulate.
    # Therefore, the minimal amount of *leader data transfer* occurs when we
    # send all the tile info to the worker and allow the worker to insert the
    # tiles into Clickhouse.
    return await backend.submit_tiles(packet_df), report


async def wait_for_packet(backend, algo, awaitable, report):
    results_df, sim_report = await backend.wait_for_results(awaitable)
    # The completion_step column could be set on the worker but that would
    # increase data transfer costs.
    report.update(sim_report)
    report["runtime_per_sim_ns"] = (
        report["runtime_simulating"] / report["n_total_sims"] * 1e9
    )
    results_df["completion_step"] = MAX_STEP
    algo.db.insert_results(results_df, algo.get_orderer())
    status = WorkerStatus.WORKING
    report["status"] = status.name
    report["runtime_total"] = time.time() - report["start_time"]
    report["done_time"] = time.time()
    algo.callback(report, algo.db)
    return report


def new_step(algo, basal_step_id, new_step_id):
    start = time.time()
    status, tiles_df, report = _new_step(algo, basal_step_id, new_step_id)
    if tiles_df is not None:
        report["n_packets"] = (
            tiles_df["packet_id"].nunique() if tiles_df is not None else 0
        )
    report["start_time"] = start
    report["status"] = status.name
    report["basal_step_id"] = basal_step_id
    report["step_id"] = new_step_id
    report["runtime_total"] = time.time() - report["start_time"]
    algo.callback(report, algo.db)
    algo.db.insert_reports(report)
    return status, tiles_df


def _new_step(algo, basal_step_id, new_step_id):
    converged, convergence_data, report = algo.convergence_criterion(basal_step_id)

    start = time.time()
    report["runtime_convergence_criterion"] = time.time() - start
    if converged:
        logger.debug("Convergence!!")
        return WorkerStatus.CONVERGED, None, report
    elif new_step_id >= algo.cfg["n_steps"]:
        logger.error(
            "Reached maximum number of steps. Terminating."
            " This should've been prevented in the outer loop."
        )
        assert False

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
    selection_df, selection_report = algo.select_tiles(
        basal_step_id, new_step_id, convergence_data
    )
    report.update(selection_report)

    if selection_df is None:
        # New step is empty so we have terminated but
        # failed to converge.
        logger.debug("New step is empty despite failure to converge.")
        return WorkerStatus.EMPTY_STEP, None, report

    selection_df["active"] = ~(selection_df["refine"] | selection_df["deepen"])

    # NOTE: this is a pathway towards eventually having variable splitting
    # logic?
    dim = ip.Grid(selection_df, None).d
    selection_df["refine"] = selection_df["refine"].astype(int) * (2**dim)
    selection_df["deepen"] = selection_df["deepen"].astype(int) * 2

    if "split" not in selection_df.columns:
        selection_df["split"] = False
    done_cols = [
        "packet_id",
        "id",
        "active",
        "refine",
        "deepen",
        "split",
    ]
    done_df = selection_df[done_cols].copy()
    done_df.insert(0, "step_id", new_step_id)
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
    g_new = refine_and_deepen(selection_df, algo.null_hypos, algo.cfg["max_K"])
    g_new.df["step_id"] = new_step_id

    # there might be new inactive tiles that resulted from splitting with
    # the null hypotheses. we need to mark these tiles as finished.
    inactive_df = g_new.df[~g_new.df["active"]].copy()
    inactive_df["packet_id"] = np.int32(-1)
    inactive_df["inactivation_step"] = new_step_id
    inactive_done = inactive_df.copy()
    inactive_done["refine"] = 0
    inactive_done["deepen"] = 0
    inactive_done["split"] = True
    inactive_done = inactive_done[["step_id"] + done_cols].copy()
    algo.db.insert_tiles(inactive_df.drop("active", axis=1), ch_insert=True)
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
    g_active.df["inactivation_step"] = MAX_STEP
    g_active.df.drop("active", axis=1, inplace=True)

    algo.db.insert_tiles(g_active.df, ch_insert=False)

    report["time"] = time.time()
    report["n_new_tiles"] = g_active.n_tiles

    logger.debug(
        f"Starting step {new_step_id}" f" with {g_active.n_tiles} tiles to simulate."
    )
    return WorkerStatus.NEW_STEP, g_active.df, report


def refine_and_deepen(df, null_hypos, max_K):
    g_deepen_in = ip.Grid(df.loc[(df["deepen"] > 0) & (df["K"] < max_K)])
    g_deepen = ip.grid._raw_init_grid(
        g_deepen_in.get_theta(),
        g_deepen_in.get_radii(),
        parents=g_deepen_in.df["id"],
    )

    # We just multiply K by 2 to deepen.
    # TODO: it's possible to do better by multiplying by 4 or 8
    # sometimes when a tile clearly needs *way* more sims. how to
    # determine this?
    g_deepen.df["K"] = g_deepen_in.df["K"] * 2

    g_refine_in = ip.grid.Grid(df.loc[df["refine"] > 0])
    inherit_cols = ["K"]
    # TODO: it's possible to do better by refining by more than just a
    # factor of 2.
    g_refine = g_refine_in.refine(inherit_cols)

    # NOTE: Instead of prune_alternative here, we mark alternative tiles as
    # inactive. This means that we will have a full history of grid
    # construction.
    out = g_refine.concat(g_deepen)
    out = out.add_null_hypos(null_hypos, inherit_cols)
    out.df.loc[out._which_alternative(), "active"] = False
    return out


class WorkerStatus(Enum):
    # Statuses which terminate the worker.
    CONVERGED = 0
    # Statuses which end the self-help stage.
    EMPTY_STEP = 2
    NO_NEW_TILES = 3
    # Normal solo work statuses.
    NEW_STEP = 4
    WORKING = 5
    ALREADY_EXISTS = 6
    EMPTY_PACKET = 7
