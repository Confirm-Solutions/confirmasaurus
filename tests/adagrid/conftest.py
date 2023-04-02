import uuid

import pytest

from confirm.adagrid.db import DuckDBTiles


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    This is used to set a report attribute for each phase of a call, which can
    be "setup", "call", "teardown" giving the request object an attribute:
    request.node.rep_call.failed which is True if the call phase failed.

    """
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"

    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture()
def ch_db(request):
    if not request.config.getoption("--run-slow"):
        pytest.skip("skipping clickhouse tests because --run-slow was not specified")

    import confirm.cloud.clickhouse as ch

    job_name = "unnamed_" + uuid.uuid4().hex
    db = ch.ClickhouseTiles.connect(job_name=job_name)
    yield db

    if not hasattr(request.node, "rep_call"):
        should_cleanup = True
    else:
        should_cleanup = not request.node.rep_call.failed

    if should_cleanup:
        db.close()
        if not request.config.option.keep_clickhouse:
            ch.clear_dbs(ch.get_ch_client(), names=[db.job_name], yes=True)
        else:
            print(
                f"Keeping Clickhouse database {db.job_name}"
                " because --keep-clickhouse was specified."
            )
    else:
        print(f"Keeping Clickhouse database {db.job_name} because test failed")


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
