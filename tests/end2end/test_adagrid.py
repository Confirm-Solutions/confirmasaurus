from unittest import mock

import numpy as np
import pandas as pd
import pytest

import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D


@pytest.mark.slow
@mock.patch("imprint.timer._timer", ip.timer.new_mock_timer())
def test_adagrid(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, db = ada.ada_calibrate(ZTest1D, g=g, nB=5, tile_batch_size=1)
    lamss = reports[-1]["lamss"]
    np.testing.assert_allclose(lamss, snapshot(lamss))
    assert iter == snapshot(iter)

    all_tiles_df = db.get_all()
    time_cols = ["id", "parent_id", "birthtime"]
    pd.testing.assert_frame_equal(
        all_tiles_df.drop(time_cols, axis=1),
        snapshot(all_tiles_df).drop(time_cols, axis=1),
        check_dtype=False,
        check_like=True,
    )

    # The second check is to make sure that the snapshot is deterministic. This
    # helps separate failures due to timing and failures due to other tile
    # quantities.
    pd.testing.assert_frame_equal(
        all_tiles_df,
        snapshot(all_tiles_df),
        check_dtype=False,
    )

    # Compare DuckDB against pandas
    pd_db = ada.db.PandasTiles()
    _, _, db2 = ada.ada_calibrate(ZTest1D, g=g, db=pd_db, nB=5, tile_batch_size=1)
    pd.testing.assert_frame_equal(
        db2.get_all().drop(["id", "parent_id"], axis=1),
        db2.get_all().drop(["id", "parent_id"], axis=1),
        check_exact=True,
    )


@pytest.mark.slow
def test_adagrid_clickhouse(snapshot):
    import confirm.cloud.clickhouse as ch

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])

    db = ch.Clickhouse.connect()
    iter, reports, db = ada.ada_calibrate(ZTest1D, g=g, db=db, nB=5, tile_batch_size=1)
    lamss = reports[-1]["lamss"]
    np.testing.assert_allclose(lamss, snapshot(lamss))


@pytest.mark.slow
def test_adagrid_checkpointing():
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ada.db.DuckDBTiles.connect()
        iter_two1, reports_two1, db_two1 = ada.ada_calibrate(
            ZTest1D, g=g, db=db, nB=3, init_K=4, n_iter=2
        )

        iter_two2, reports_two2, db_two2 = ada.ada_calibrate(ZTest1D, db=db, n_iter=1)
        reports_two2[0]["i"] = 2

    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db_once = ada.ada_calibrate(
            ZTest1D, g=g, nB=3, init_K=4, n_iter=3
        )

    assert iter_two1 + iter_two2 == iter
    drop_cols = [c for c in reports[0].keys() if c.startswith("runtime")]
    pd.testing.assert_frame_equal(
        pd.DataFrame(reports_two1 + reports_two2).drop(drop_cols, axis=1),
        pd.DataFrame(reports).drop(drop_cols, axis=1),
    )
    nondeterministic_cols = ["birthiter", "worker_id"]
    df_once = db_once.get_all()
    df_twice = db_two2.get_all()
    pd.testing.assert_frame_equal(
        df_twice.drop(nondeterministic_cols, axis=1),
        df_once.drop(nondeterministic_cols, axis=1),
    )

    second_phase_ids_twice = set(
        df_twice.loc[df_twice["worker_id"] == 1, "id"].tolist()
    )
    second_phase_ids_once = set(df_once.loc[df_once["birthiter"] == 2, "id"].tolist())
    assert second_phase_ids_once == second_phase_ids_twice


def main():
    import confirm.cloud.clickhouse as ch

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ch.Clickhouse.connect()
    # db = ch.Clickhouse.connect(
    #     host='localhost', port='8123', username='default', password='')
    iter, reports, db = ada.ada_calibrate(
        ZTest1D, db=db, g=g, nB=5, tile_batch_size=1, n_iter=8
    )


if __name__ == "__main__":
    main()
