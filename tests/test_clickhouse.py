import pytest
import numpy as np
import imprint as ip

from test_db import DBTester
from test_store import StoreTester

import confirm.cloud.clickhouse as ch


def test_ch_not_slow():
    g = ip.cartesian_grid([-1, -2], [1, 2], n=[3, 3])
    g.df["step_id"] = 1
    g.df["step_iter"] = 1
    db = ch.Clickhouse.connect()
    db.init_tiles(g.df)
    np.testing.assert_allclose(db.get_tiles()["theta0"], g.df["theta0"])


@pytest.mark.slow
class TestClickhouse(DBTester):
    def setup_class(cls):
        cls.dbtype = ch.Clickhouse

    def test_connect(self):
        self.dbtype.connect()


@pytest.mark.slow
class TestClickhouseStore(StoreTester):
    def connect(self):
        return ch.Clickhouse.connect().store


@pytest.mark.slow
def test_connect_prod_no_job_id():
    with pytest.raises(RuntimeError) as excinfo:
        ch.Clickhouse.connect(host="fakeprod")
    assert "To run a production job" in str(excinfo.value)


@pytest.mark.slow
def test_clear_dbs_only_test():
    class FakeClient:
        def __init__(self):
            self.url = "fakeprod"

    with pytest.raises(RuntimeError) as excinfo:
        ch.clear_dbs(FakeClient(), None)
    assert "localhost" in str(excinfo.value) and "test" in str(excinfo.value)
