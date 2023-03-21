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

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ada.ada_calibrate(
        ZTest1D,
        g=g,
        init_K=1024,
        nB=2,
        grid_target=0.005,
        bias_target=0.005,
        std_target=0.005,
        prod=False,
        tile_batch_size=1,
        backup_interval=1,
        job_name=ch_db.database,
    )

    db2 = asyncio.run(ch.restore(ch_db.database))

    for table in ch.all_tables:
        print(table)
        orig = db.con.query(f"select * from {table}").df()
        restored = db2.con.query(f"select * from {table}").df()
        pd.testing.assert_frame_equal(orig, restored)
