import asyncio
import logging

import pandas as pd

from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.init import _launch_task
from confirm.adagrid.init import assign_tiles

logger = logging.getLogger(__name__)


async def coordinate(algo, step_id, n_zones):
    report = dict()
    status, lazy_tasks = await _coordinate(algo, step_id, n_zones, report)
    report["status"] = status.name
    algo.callback(report, algo.db)
    insert_report = await _launch_task(algo.db, algo.db.insert_report, report)
    lazy_tasks.append(insert_report)
    return status, lazy_tasks


async def _coordinate(algo, step_id, n_zones, report):
    # This function should only ever run for the clickhouse database. So,
    # rather than abstracting the various database calls, we just write
    # them in here directly.
    import confirm.cloud.clickhouse as ch

    report["step_id"] = step_id
    report["worker_id"] = algo.cfg["worker_id"]
    report["n_zones"] = n_zones

    alter_settings = {
        "allow_nondeterministic_mutations": "1",
        "mutations_sync": 2,
    }
    # This can happen asynchronously because:
    # 1. An inactive tile or ineligible tile will never revert to being active
    #    or eligible, respectively.
    # 2. The done and inactive tables are append-only.
    # 3. All queries that depend on active and eligible use the and(...) anyway.
    # So, it's not necessary to update eligible and active but it will make future
    # queries faster.
    update_active_eligible_task = await _launch_task(
        algo.db,
        algo.db.client.command,
        """
        ALTER TABLE results
        UPDATE
            eligible = and(eligible=true, id not in (select id from done)),
            active = and(active=true, id not in (select id from inactive))
        WHERE
            eligible = 1
            and active = 1
        """,
        settings=alter_settings,
    )

    converged, _ = algo.convergence_criterion(None, report)
    if converged:
        return WorkerStatus.CONVERGED

    df = ch._query_df(
        algo.db.client,
        """
        SELECT * FROM results
        WHERE eligible = 1
            and id not in (select id from done)
            and active = 1
            and id not in (select id from inactive)
        """,
    )
    report["n_tiles"] = df.shape[0]
    if df.shape[0] == 0:
        return WorkerStatus.EMPTY_STEP, [update_active_eligible_task]

    df["eligible"] = True
    df["active"] = True
    old_zone_id = df["zone_id"].values.copy()
    new_zone_id = assign_tiles(df.shape[0], n_zones)
    df["zone_id"] = new_zone_id
    old_coordination_id = df["coordination_id"].values[0]
    assert (df["coordination_id"] == old_coordination_id).all()
    df["coordination_id"] += 1

    insert_task = await _launch_task(
        algo.db, ch._insert_df, algo.db.client, "results", df
    )
    delete_task = await _launch_task(
        algo.db,
        algo.db.client.command,
        f"""
        ALTER TABLE results
        DELETE WHERE eligible = 1
                and id not in (select id from done)
                and active = 1
                and id not in (select id from inactive)
                and coordination_id = {old_coordination_id}
        """,
        settings=alter_settings,
    )

    def insert_mapping(old_zone_id, df):
        if not ch.does_table_exist(algo.db.client, algo.db.job_id, "zone_mapping"):
            id_type = ch.type_map[df["id"].dtype.name]
            algo.db.client.command(
                f"""
                CREATE TABLE zone_mapping
                (
                    id {id_type},
                    old_zone_id Int32,
                    new_zone_id Int32
                )
                ENGINE = MergeTree()
                ORDER BY (old_zone_id, new_zone_id)
                """
            )
        mapping_df = pd.DataFrame(
            {"id": df["id"], "old_zone_id": old_zone_id, "new_zone_id": df["zone_id"]}
        )
        ch._insert_df(algo.db.client, "zone_mapping", mapping_df)

    insert_mapping_task = await _launch_task(algo.db, insert_mapping, old_zone_id, df)
    await asyncio.gather(insert_task, delete_task)

    return WorkerStatus.COORDINATED, [insert_mapping_task, update_active_eligible_task]
