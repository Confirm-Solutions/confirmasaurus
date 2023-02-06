import pytest


@pytest.fixture()
def ch_db():
    import confirm.cloud.clickhouse as ch

    db = ch.Clickhouse.connect()
    yield db
    db.close()
    ch.clear_dbs(ch.get_ch_client(), None, names=[db.job_id], yes=True)
