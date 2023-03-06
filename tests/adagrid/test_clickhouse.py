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

    def _connect(self, no_async=False):
        import confirm.cloud.clickhouse as ch

        self.dbs.append(ch.Clickhouse.connect())
        if no_async:
            self.dbs[-1].async_insert_settings = ch.default_insert_settings
        return self.dbs[-1]


@pytest.mark.slow
class TestClickhouse(DBTester, ClickhouseCleanup):
    def connect(self, no_async=False):
        return self._connect(no_async=no_async)


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
