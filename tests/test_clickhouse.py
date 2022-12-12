import pytest
from test_duckdb import DBTester


@pytest.mark.slow
class TestClickhouse(DBTester):
    def setup_class(cls):
        from confirm.cloud.clickhouse import Clickhouse

        cls.dbtype = Clickhouse

    def test_connect(self):
        self.dbtype.connect()
