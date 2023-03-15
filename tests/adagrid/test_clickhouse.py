import pandas as pd
import pytest

from ..test_db import DBTester


class ClickhouseCleanup:
    def setup_method(self, method):
        self.dbs = []

    def teardown_method(self, method):
        import confirm.cloud.clickhouse as ch

        client = ch.get_ch_client()
        job_ids = [db.job_id for db in self.dbs]
        for db in self.dbs:
            db.close()
        ch.clear_dbs(client, names=job_ids, yes=True)

    def _connect(self):
        import confirm.cloud.clickhouse as ch

        self.dbs.append(ch.Clickhouse.connect())
        return self.dbs[-1]


@pytest.mark.slow
class TestClickhouse(DBTester, ClickhouseCleanup):
    def connect(self):
        return self._connect()


@pytest.mark.slow
def test_connect_prod_no_job_id():
    import confirm.cloud.clickhouse as ch

    with pytest.raises(RuntimeError) as excinfo:
        ch.Clickhouse.connect(host="fakeprod")
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
        coordinate_every=1,
        n_zones=2,
    )

    ch.backup(db, ch_db)
    db2 = ada.DuckDBTiles.connect()
    ch.restore(db2, ch_db)
    for table in ch.all_tables:
        if not db.does_table_exist(table):
            continue
        orig = db.con.query(f"select * from {table}").df()
        restored = db2.con.query(f"select * from {table}").df()
        pd.testing.assert_frame_equal(orig, restored)
