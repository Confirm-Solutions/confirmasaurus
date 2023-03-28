import asyncio

import pandas as pd
import pytest


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
    import imprint as ip
    import confirm.adagrid as ada
    import confirm.cloud.clickhouse as ch
    from imprint.models.ztest import ZTest1D

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
        print(table)
        orig = db.con.query(f"select * from {table}").df()
        restored = db2.con.query(f"select * from {table}").df()
        if table in orderby:
            orig = orig.sort_values(by=orderby[table]).reset_index(drop=True)
            restored = restored.sort_values(by=orderby[table]).reset_index(drop=True)
        pd.testing.assert_frame_equal(orig, restored, check_dtype=False)
