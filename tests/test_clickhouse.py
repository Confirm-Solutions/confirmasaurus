import pytest
from test_duckdb import DBTester

from confirm.cloud.clickhouse import Clickhouse


@pytest.mark.slow
class TestClickhouse(DBTester):
    dbtype = Clickhouse

    def test_connect(self):
        Clickhouse.connect()
