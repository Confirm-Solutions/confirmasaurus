import pytest
from test_db import DBTester
from test_store import StoreTester

import confirm.cloud.clickhouse as ch

class ClickhouseCleanup:
    def setup_method(self, method):
        self.dbs = []
    
    def teardown_method(self, method):
        client = ch.get_ch_client()
        job_ids = [db.job_id for db in self.dbs]
        for db in self.dbs:
            db.close()
        ch.clear_dbs(client, None, names=job_ids, yes=True)
        
    def _connect(self):
        self.dbs.append(ch.Clickhouse.connect())
        return self.dbs[-1]

@pytest.mark.slow
class TestClickhouse(DBTester, ClickhouseCleanup):
    def connect(self):
        return self._connect()

@pytest.mark.slow
class TestClickhouseStore(StoreTester, ClickhouseCleanup):
    def connect(self):
        return self._connect().store


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
