import asyncio
import logging

from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.init import _launch_task
from confirm.adagrid.init import assign_tiles

logger = logging.getLogger(__name__)


async def coordinate(algo, step_id, n_zones):
    report = dict()
    status, lazy_tasks, zone_steps = await _coordinate(algo, step_id, n_zones, report)
    report["status"] = status.name
    algo.callback(report, algo.db)
    insert_report = await _launch_task(algo.db, algo.db.insert_report, report)
    lazy_tasks.append(insert_report)
    return status, lazy_tasks, zone_steps


async def _coordinate(algo, step_id, n_zones, report):
    # This function should only ever run for the clickhouse database. So,
    # rather than abstracting the various database calls, we just write
    # them in here directly.
    report["step_id"] = step_id
    report["worker_id"] = algo.cfg["worker_id"]
    report["n_zones"] = n_zones

    # Updating active/eligible state can happen asynchronously because:
    # 1. An inactive tile or ineligible tile will never revert to being active
    #    or eligible, respectively.
    # 2. The done and inactive tables are append-only.
    # 3. All queries that depend on active and eligible use the and(...) anyway.
    # So, it's not necessary to update eligible and active but it will make future
    # queries faster.
    lazy_tasks = [
        await _launch_task(
            algo.db,
            algo.db.update_active_eligible,
        )
    ]

    converged, _ = algo.convergence_criterion(None, report)
    if converged:
        return WorkerStatus.CONVERGED, lazy_tasks, None

    df = algo.db.get_active_eligible()
    report["n_tiles"] = df.shape[0]
    if df.shape[0] == 0:
        return WorkerStatus.EMPTY_STEP, lazy_tasks, None

    df["eligible"] = True
    df["active"] = True
    old_zone_id = df["zone_id"].values.copy()
    new_zone_id = assign_tiles(df.shape[0], n_zones)
    df["zone_id"] = new_zone_id
    zone_steps = {i: step_id for i, zone in df.groupby("zone_id")}
    old_coordination_id = df["coordination_id"].values[0]
    assert (df["coordination_id"] == old_coordination_id).all()
    df["coordination_id"] += 1

    insert_task = await _launch_task(
        algo.db, algo.db.insert_results, df, algo.get_orderer()
    )
    delete_task = await _launch_task(
        algo.db, algo.db.delete_previous_coordination, old_coordination_id
    )
    mapping_df = df[["id", "coordination_id", "zone_id"]].copy()
    mapping_df["old_zone_id"] = old_zone_id
    mapping_df["before_step_id"] = step_id
    insert_mapping_task = await _launch_task(
        algo.db, algo.db.insert_mapping, mapping_df
    )
    await asyncio.gather(insert_task, delete_task)

    lazy_tasks.append(insert_mapping_task)
    return WorkerStatus.COORDINATED, lazy_tasks, zone_steps
