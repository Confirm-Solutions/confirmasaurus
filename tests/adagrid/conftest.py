import pytest

from confirm.adagrid.db import DuckDBTiles


@pytest.fixture()
def ch_db(request):
    import confirm.cloud.clickhouse as ch

    db = ch.Clickhouse.connect()
    yield db
    print(f"Cleaning up Clickhouse database {db.job_id}")
    db.close()
    if not request.config.option.keep_clickhouse:
        ch.clear_dbs(ch.get_ch_client(), None, names=[db.job_id], yes=True)


@pytest.fixture
def duckdb():
    return DuckDBTiles.connect()


@pytest.fixture(params=["duckdb", "ch_db"])
def both_dbs(request):
    return request.getfixturevalue(request.param)


def pytest_addoption(parser):
    parser.addoption(
        "--keep-clickhouse",
        action="store_true",
        default=False,
        dest="keep_clickhouse",
        help="Don't delete Clickhouse databases.",
    )
