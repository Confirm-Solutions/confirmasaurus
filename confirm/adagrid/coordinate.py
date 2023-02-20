import logging

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
        return WorkerStatus.EMPTY_STEP

    df["eligible"] = True
    df["active"] = True
    df["zone_id"] = assign_tiles(df.shape[0], n_zones)
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

    return WorkerStatus.COORDINATED
