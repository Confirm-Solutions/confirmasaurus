import pytest
from test_duckdb import DBTester


@pytest.mark.slow
class TestClickhouse(DBTester):
    from confirm.cloud.clickhouse import Clickhouse

    dbtype = Clickhouse

    def test_connect(self):
        self.dbtype.connect()
