import asyncio
import time

import pandas as pd
import pytest

import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D


def check_backup_correctness(db):
    import confirm.cloud.clickhouse as ch

    # wait for all the asynchronous CH inserts to finish
    n_rows_df = pd.DataFrame(
        [
            (table, db.con.query(f"select count(*) from {table}").df().iloc[0][0])
            for table in ch.all_tables
        ],
        columns=["table", "count"],
    ).set_index("table")
    i = 0
    while i < 60:
        time.sleep(1)
        n_rows_ch = pd.DataFrame(
            [
                (
                    table,
                    ch.query(db.ch_client, f"select count(*) from {table}").result_set[
                        0
                    ][0],
                )
                for table in ch.all_tables
            ],
            columns=["table", "count"],
        ).set_index("table")
        if (n_rows_df == n_rows_ch).all().all():
            break
        else:
            print("Waiting for asynchronous Clickhouse inserts to finish.")
        i += 1
    else:
        raise TimeoutError("Clickhouse results not restored")

    db2 = ada.DuckDBTiles.connect()
    asyncio.run(ch.restore(db.ch_client, db2))

    orderby = {
        "results": "id",
        "tiles": "id",
        "done": "id",
        "logs": "t",
        "reports": "json",
    }
    for table in ch.all_tables:
        orig = db.con.query(f"select * from {table}").df()
        restored = db2.con.query(f"select * from {table}").df()
        if table in orderby:
            orig = orig.sort_values(by=orderby[table]).reset_index(drop=True)
            restored = restored.sort_values(by=orderby[table]).reset_index(drop=True)
        pd.testing.assert_frame_equal(orig, restored[orig.columns], check_dtype=False)


@pytest.mark.slow
def test_connect_prod_no_job_id():
    import confirm.cloud.clickhouse as ch

    with pytest.raises(RuntimeError) as excinfo:
        ch.connect(host="fakeprod")
    assert "To run a production job" in str(excinfo.value)


@pytest.mark.slow
def test_clear_dbs_only_test():
    import confirm.cloud.clickhouse as ch

    class FakeClient:
        def __init__(self):
            self.url = "fakeprod"

    with pytest.raises(RuntimeError) as excinfo:
        ch.clear_dbs(FakeClient(), None)
    assert "localhost" in str(excinfo.value) and "test" in str(excinfo.value)


@pytest.mark.slow
def test_backup(ch_db):
    import confirm.cloud.clickhouse as ch

    # set backups to happen synchronously
    ch.set_insert_settings(ch.synchronous_insert_settings)

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ada.DuckDBTiles.connect()
    ada.ada_calibrate(
        ZTest1D,
        g=g,
        db=db,
        init_K=1024,
        nB=2,
        grid_target=0.005,
        bias_target=0.005,
        std_target=0.005,
        record_system=False,
        clickhouse_service="TEST",
        job_name=ch_db.database,
        tile_batch_size=1,
    )
    check_backup_correctness(db)
