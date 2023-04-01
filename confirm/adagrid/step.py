import asyncio
import logging
import time
from enum import Enum

import numpy as np
import pandas as pd

import imprint as ip

logger = logging.getLogger(__name__)


async def process_initial_packets(backend, algo, tiles_df):
    wait_for = []
    for packet_id, packet_df in tiles_df.groupby("packet_id"):
        wait_for.append(
            await submit_packet(backend, algo, packet_df, refine_deepen=False)
        )
    await wait_for_packets(backend, algo, wait_for)


async def submit_packet_df(backend, algo, tiles_df):
    if tiles_df is None or tiles_df.shape[0] == 0:
        return []
    packets = []
    for packet_id, packet_df in tiles_df.groupby("packet_id"):
        packets.append(
            await submit_packet(backend, algo, packet_df, refine_deepen=True)
        )
    return packets


async def wait_for_packets(backend, algo, packets):
    coros = []
    for p in packets:
        coros.append(wait_for_packet(backend, algo, *p))
    logger.debug("Waiting for packets to finish.")
    reports = await asyncio.gather(*coros)
    logger.debug("Packets finished.")
    algo.db.insert_reports(reports)


async def submit_packet(backend, algo, packet_df, refine_deepen):
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
    return await backend.submit_tiles(packet_df, refine_deepen), report


async def wait_for_packet(backend, algo, awaitable, report):
    sim_report = await backend.wait_for_results(awaitable)
    # TODO: move to worker?
    report.update(sim_report)
    report["runtime_per_sim_ns"] = (
        report["runtime_simulating"] / report["n_total_sims"] * 1e9
    )
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

    def assign_packets(df):
        return pd.Series(
            np.floor(np.arange(df.shape[0]) / algo.cfg["packet_size"]).astype(int),
            df.index,
        )

    selection_df["packet_id"] = assign_packets(selection_df)
    selection_df["step_id"] = new_step_id
    algo.db.insert_done(selection_df)

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
    return WorkerStatus.NEW_STEP, selection_df, report


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
