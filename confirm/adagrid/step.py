import asyncio
import logging
import time
from enum import Enum

import jax
import pandas as pd

import imprint as ip
from .const import MAX_STEP

logger = logging.getLogger(__name__)


async def submit_packet_df(backend, algo, tiles_df, refine_deepen: bool = True):
    async def submit_packet(packet_df):
        start = time.time()
        report = dict()
        report["start_time"] = start
        logger.debug("Submitting %d tiles for processing.", packet_df.shape[0])
        return await backend.submit_tiles(packet_df, refine_deepen, report)

    if tiles_df is None or tiles_df.shape[0] == 0:
        return []
    packets = []
    for packet_id, packet_df in tiles_df.groupby("packet_id"):
        packets.append(await submit_packet(packet_df))
    return packets


async def wait_for_packets(backend, algo, packets):
    async def wait_for_packet(awaitable):
        n_inserts, report = await backend.wait_for_results(awaitable)
        # TODO: move to worker?
        report["runtime_total"] = time.time() - report["start_time"]
        report["done_time"] = time.time()
        algo.callback(report, algo.db)
        return n_inserts, report

    coros = []
    for p in packets:
        coros.append(wait_for_packet(p))
    logger.debug("Waiting for packets to finish.")
    outs = await asyncio.gather(*coros)
    if len(outs) == 0:
        n_inserts = [dict(tiles=0, results=0, done=0)]
        reports = []
    else:
        n_inserts, reports = zip(*outs)
    logger.debug("Packets finished.")
    algo.db.insert_reports(*reports)
    out_n_inserts = dict(tiles=0, results=0, done=0)
    for n in n_inserts:
        for k in n:
            out_n_inserts[k] += n[k]
    return out_n_inserts


def process_tiles(algo, df, refine_deepen: bool, report: dict):
    if refine_deepen:
        start = time.time()
        tiles_df, inactive_df = refine_and_deepen(
            df, algo.null_hypos, algo.cfg["max_K"]
        )
        algo.db.insert("tiles", inactive_df)
        algo.db.insert("done", inactive_df)
        report["runtime_refine_deepen"] = time.time() - start
        n_inactive = inactive_df.shape[0]
    else:
        tiles_df = df
        n_inactive = 0

    algo.db.insert("tiles", tiles_df)

    tbs = algo.cfg["tile_batch_size"]
    if tbs is None:
        tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
    report["sim_start_time"] = time.time()
    results_df = algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)
    report["sim_done_time"] = time.time()
    results_df["completion_step"] = MAX_STEP
    results_df["inactivation_step"] = MAX_STEP
    algo.db.insert("results", results_df)

    report["step_id"] = df.iloc[0]["step_id"]
    report["packet_id"] = df.iloc[0]["packet_id"]
    report["n_parent_tiles"] = df.shape[0]
    report["n_tiles"] = tiles_df.shape[0]
    report["n_inactive_tiles"] = n_inactive
    report["tile_batch_size"] = tbs
    report["n_total_sims"] = results_df["K"].sum()
    report["runtime_simulating"] = report["sim_done_time"] - report["sim_start_time"]
    report["runtime_per_sim_ns"] = (
        report["runtime_simulating"] / report["n_total_sims"] * 1e9
    )
    report["status"] = WorkerStatus.WORKING.name

    n_inserts = dict(
        tiles=tiles_df.shape[0] + n_inactive,
        done=n_inactive,
        results=results_df.shape[0],
    )
    return n_inserts, report


def refine_and_deepen(df, null_hypos, max_K):
    g_deepen_in = ip.Grid(df.loc[(df["deepen"] > 0) & (df["K"] < max_K)])
    g_deepen = ip.grid._raw_init_grid(
        g_deepen_in.get_theta(),
        g_deepen_in.get_radii(),
        parents=g_deepen_in.df["id"],
    )
    # No need to recalculate null_truth after deepening because the tile
    # positions have not changed.
    for i in range(len(null_hypos)):
        g_deepen.df[f"null_truth{i}"] = g_deepen_in.df[f"null_truth{i}"]
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
    g_refine = g_refine.add_null_hypos(null_hypos, inherit_cols)
    g_refine.df.loc[g_refine._which_alternative(), "active"] = False

    g_new = g_refine.concat(g_deepen)
    for col in ["step_id", "packet_id"]:
        g_new.df[col] = df[col].iloc[0]
    g_new.df.rename(columns={"active": "active_at_birth"}, inplace=True)
    inactive_df = g_new.df[~g_new.df["active_at_birth"]].copy()
    inactive_df["refine"] = 0
    inactive_df["deepen"] = 0
    inactive_df["split"] = 1
    inactive_df["active"] = False

    active_df = g_new.df[g_new.df["active_at_birth"]]
    return active_df, inactive_df


@profile
def new_step(algo, basal_step_id, new_step_id):
    start = time.time()
    status, tiles_df, report, n_inserts = _new_step(algo, basal_step_id, new_step_id)
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
    for k in ["tiles", "results", "done"]:
        if k not in n_inserts:
            n_inserts[k] = 0
    return status, tiles_df, n_inserts


@profile
def _new_step(algo, basal_step_id, new_step_id):
    converged, convergence_data, report = algo.convergence_criterion(basal_step_id)

    start = time.time()
    report["runtime_convergence_criterion"] = time.time() - start
    if converged:
        logger.debug("Convergence!!")
        return WorkerStatus.CONVERGED, None, report, {}
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
            {},
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
        return WorkerStatus.EMPTY_STEP, None, report, {}

    selection_df["active"] = ~(selection_df["refine"] | selection_df["deepen"])

    # NOTE: this is a pathway towards eventually having variable splitting
    # logic?
    dim = ip.Grid(selection_df, None).d
    selection_df["refine"] = selection_df["refine"].astype(int) * (2**dim)
    selection_df["deepen"] = selection_df["deepen"].astype(int) * 2

    if "split" not in selection_df.columns:
        selection_df["split"] = False

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
        return WorkerStatus.NO_NEW_TILES, None, report, {}

    selection_df["packet_id"] = assign_packets(selection_df, algo.cfg["packet_size"])
    selection_df["step_id"] = new_step_id
    algo.db.insert_done_update_results(selection_df)

    report["time"] = time.time()
    report["n_new_tiles"] = (
        selection_df["refine"].sum() + (selection_df["deepen"] > 0).sum()
    )
    report["n_new_sims"] = (
        selection_df["K"] * (selection_df["refine"] + selection_df["deepen"])
    ).sum()
    logger.debug(
        f"Starting step {new_step_id}"
        f" with {report['n_new_tiles']} tiles to simulate."
    )
    n_inserts = dict(tiles=0, done=selection_df.shape[0], results=0)
    return WorkerStatus.NEW_STEP, selection_df, report, n_inserts


def assign_packets(df, packet_size):
    cum_sims = (df["K"] * (df["refine"] + df["deepen"])).cumsum()
    return pd.Series((cum_sims // packet_size).astype(int), df.index)


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
