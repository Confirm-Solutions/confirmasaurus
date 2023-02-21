import logging

import pandas as pd

from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.init import _launch_task
from confirm.adagrid.init import assign_tiles

logger = logging.getLogger(__name__)


async def coordinate(algo, step_id, n_zones):
    report = dict()
    status = await _coordinate(algo, report)
    report["status"] = status.name
    algo.callback(report, algo.db)
    insert_report = await _launch_task(algo.db, algo.db.insert_report, report)
    return status, insert_report


async def _coordinate(algo, step_id, n_zones, report):
    # This function should only ever run for the clickhouse database. So,
    # rather than abstracting the various database calls, we just write
    # them in here directly.
    import confirm.cloud.clickhouse as ch

    report["step_id"] = step_id
    report["worker_id"] = algo.cfg["worker_id"]
    report["n_zones"] = n_zones

    # It's not necessary to update eligible and active but it will make future
    # queries faster.
    alter_settings = {
        "allow_nondeterministic_mutations": "1",
        "mutations_sync": 2,
    }
    # TODO: this could be async?
    algo.db.client.command(
        """
        ALTER TABLE results
        UPDATE
            eligible = and(eligible=true, id not in (select id from done)),
            active = and(active=true, id not in (select id from inactive))
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
        return WorkerStatus.EMPTY_STEP, []

    df["eligible"] = True
    df["active"] = True
    old_zone_id = df["zone_id"].values.copy()
    new_zone_id = assign_tiles(df.shape[0], n_zones)
    df["zone_id"] = new_zone_id
    algo.db.client.command(
        """
        ALTER TABLE results
        DELETE WHERE eligible = 1
                and id not in (select id from done)
                and active = 1
                and id not in (select id from inactive)
        """,
        settings=alter_settings,
    )
    ch._insert_df(algo.db.client, "results", df)

    def insert_mapping(old_zone_id, new_zone_id):
        if not ch.does_table_exist(algo.db.client, "zone_mapping"):
            algo.db.client.command(
                """
                CREATE TABLE zone_mapping
                (
                    old_zone_id Int32,
                    new_zone_id Int32
                )
                ENGINE = MergeTree()
                ORDER BY (old_zone_id, new_zone_id)
                """
            )
        mapping_df = pd.DataFrame(
            {"old_zone_id": old_zone_id, "new_zone_id": new_zone_id}
        )
        ch._insert_df(algo.db.client, "zone_mapping", mapping_df)

    insert_mapping_task = await _launch_task(
        algo.db, insert_mapping, old_zone_id, new_zone_id
    )

    return WorkerStatus.COORDINATED, [insert_mapping_task]
