from unittest import mock

import numpy as np
import pandas as pd
import pytest

import confirm.imprint as ip
from confirm.models.ztest import ZTest1D


@pytest.mark.slow
@mock.patch("confirm.imprint.grid.uuid_timer", mock.MagicMock(return_value=100))
def test_adagrid(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, db = ip.ada_calibrate(ZTest1D, g=g, nB=5, tile_batch_size=1)
    lamss = reports[-1]["lamss"]
    np.testing.assert_allclose(lamss, snapshot(lamss))
    assert iter == snapshot(iter)

    all_tiles_df = db.get_all()
    pd.testing.assert_frame_equal(
        all_tiles_df, snapshot(all_tiles_df), check_dtype=False
    )

    # Compare DuckDB against pandas
    pd_db = ip.db.PandasTiles()
    _, _, db2 = ip.ada_calibrate(ZTest1D, g=g, db=pd_db, nB=5, tile_batch_size=1)
    pd.testing.assert_frame_equal(
        db2.get_all().drop(["id", "parent_id"], axis=1),
        db2.get_all().drop(["id", "parent_id"], axis=1),
        check_exact=True,
    )


@pytest.mark.slow
@mock.patch("confirm.imprint.grid.uuid_timer", mock.MagicMock(return_value=100))
def test_adagrid_clickhouse(snapshot):
    import confirm.cloud.clickhouse as ch

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])

    db = ch.Clickhouse.connect()
    iter, reports, db = ip.ada_calibrate(ZTest1D, g=g, db=db, nB=5, tile_batch_size=1)
    lamss = reports[-1]["lamss"]
    np.testing.assert_allclose(lamss, snapshot(lamss))


@pytest.mark.slow
def test_adagrid_checkpointing():
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ip.db.DuckDBTiles.connect()
    iter_two1, reports_two1, db_two1 = ip.ada_calibrate(
        ZTest1D, g=g, db=db, nB=3, init_K=4, n_iter=2
    )

    iter_two2, reports_two2, db_two2 = ip.ada_calibrate(ZTest1D, db=db, n_iter=1)
    reports_two2[0]["i"] = 2

    iter, reports, db_once = ip.ada_calibrate(ZTest1D, g=g, nB=3, init_K=4, n_iter=3)
    assert iter_two1 + iter_two2 == iter
    drop_cols = [c for c in reports[0].keys() if c.startswith("runtime")]
    pd.testing.assert_frame_equal(
        pd.DataFrame(reports_two1 + reports_two2).drop(drop_cols, axis=1),
        pd.DataFrame(reports).drop(drop_cols, axis=1),
    )
    nondeterministic_cols = ["birthiter", "birthtime", "worker_id", "id", "parent_id"]
    pd.testing.assert_frame_equal(
        db_two2.get_all().drop(nondeterministic_cols, axis=1),
        db_once.get_all().drop(nondeterministic_cols, axis=1),
    )


def main():
    import confirm.cloud.clickhouse as ch

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ch.Clickhouse.connect()
    # db = ch.Clickhouse.connect(
    #     host='localhost', port='8123', username='default', password='')
    iter, reports, ada = ip.ada_calibrate(
        ZTest1D, db=db, g=g, nB=5, tile_batch_size=1, n_iter=8
    )


if __name__ == "__main__":
    main()
