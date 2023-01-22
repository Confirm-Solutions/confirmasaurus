import pytest
from test_db import DBTester
from test_store import StoreTester

import confirm.cloud.clickhouse as ch


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


def test_connect_prod_no_job_id():
    import confirm.cloud.clickhouse as ch

    with pytest.raises(RuntimeError) as excinfo:
        ch.Clickhouse.connect(host="fakeprod")
    assert "To run a production job" in str(excinfo.value)


def test_clear_dbs_only_test():
    class FakeClient:
        def __init__(self):
            self.url = "fakeprod"

    with pytest.raises(RuntimeError) as excinfo:
        ch.clear_dbs(FakeClient(), None)
    assert "localhost" in str(excinfo.value) and "test" in str(excinfo.value)
