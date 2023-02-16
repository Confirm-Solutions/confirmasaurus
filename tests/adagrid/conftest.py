import pytest


@pytest.fixture()
def ch_db(request):
    import confirm.cloud.clickhouse as ch

    db = ch.Clickhouse.connect()
    yield db
    db.close()
    if not request.config.option.keep_clickhouse:
        ch.clear_dbs(ch.get_ch_client(), None, names=[db.job_id], yes=True)


def pytest_addoption(parser):
    parser.addoption(
        "--keep-clickhouse",
        action="store_true",
        default=False,
        dest="keep_clickhouse",
        help="Don't delete Clickhouse databases.",
    )
