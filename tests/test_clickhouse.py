import pytest
from test_duckdb import DBTester
from test_store import StoreTester


@pytest.mark.slow
class TestClickhouse(DBTester):
    def setup_class(cls):
        from confirm.cloud.clickhouse import Clickhouse

        cls.dbtype = Clickhouse

    def test_connect(self):
        self.dbtype.connect()


@pytest.mark.slow
class TestClickhouseStore(StoreTester):
    def connect(self):
        from confirm.cloud.clickhouse import Clickhouse

        return Clickhouse.connect().store
