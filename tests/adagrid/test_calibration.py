from unittest import mock

import numpy as np
import pandas as pd
import pytest
from jax.config import config

import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D

config.update("jax_enable_x64", True)


def test_bootstrap_calibrate():
    g = ip.cartesian_grid(
        theta_min=[-1], theta_max=[1], n=[10], null_hypos=[ip.hypo("x0 < 0")]
    )
    cal_df = ada.bootstrap.bootstrap_calibrate(ZTest1D, g=g, nB=5)
    twb_cols = [c for c in cal_df.columns if "twb_lams" in c]
    np.testing.assert_allclose(cal_df["twb_mean_lams"], cal_df[twb_cols].mean(axis=1))
    np.testing.assert_allclose(cal_df["twb_min_lams"], cal_df[twb_cols].min(axis=1))
    np.testing.assert_allclose(cal_df["twb_max_lams"], cal_df[twb_cols].max(axis=1))


def test_calibration_cheap(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ada.ada_calibrate(
        ZTest1D,
        g=g,
        init_K=1024,
        nB=2,
        grid_target=0.005,
        bias_target=0.005,
        std_target=0.005,
        prod=False,
        tile_batch_size=1,
    )
    ip.testing.check_imprint_results(
        ip.Grid(db.get_results(), None).prune_inactive(), snapshot
    )


@pytest.mark.slow
def test_calibration(snapshot):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ada.ada_calibrate(ZTest1D, g=g, nB=5, prod=False, tile_batch_size=1)
    ip.testing.check_imprint_results(
        ip.Grid(db.get_results(), None).prune_inactive(), snapshot, ignore_story=False
    )

    # Compare DuckDB against pandas
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        pd_db = ada.db.PandasTiles()
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        db2 = ada.ada_calibrate(
            ZTest1D, g=g, db=pd_db, nB=5, prod=False, tile_batch_size=1
        )

    pd.testing.assert_frame_equal(
        db.get_results(),
        db2.get_results(),
        check_exact=True,
    )


@pytest.mark.slow
def test_calibration_packetsize1(snapshot):
    snapshot.set_test_name("test_calibration")
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ada.ada_calibrate(ZTest1D, g=g, nB=5, tile_batch_size=1, packet_size=1)
    ip.testing.check_imprint_results(
        ip.Grid(db.get_results(), None).prune_inactive(), snapshot
    )


@pytest.mark.slow
def test_calibration_clickhouse(snapshot, ch_db):
    snapshot.set_test_name("test_calibration")
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ada.ada_calibrate(ZTest1D, g=g, db=ch_db, nB=5, tile_batch_size=1)

    ip.testing.check_imprint_results(
        ip.Grid(db.get_results(), None).prune_inactive(), snapshot
    )


@pytest.mark.slow
def test_solo_coordinations(snapshot):
    snapshot.set_test_name("test_calibration")
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ada.ada_calibrate(
            ZTest1D,
            g=g,
            nB=5,
            tile_batch_size=1,
            backend=ada.LocalBackend(n_zones=1, coordinate_every=1),
        )

    ip.testing.check_imprint_results(
        ip.Grid(db.get_results(), None).prune_inactive(), snapshot
    )


def four_zones_tester(db, snapshot, backend=None):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        if backend is None:
            backend = ada.LocalBackend(n_zones=4, coordinate_every=1)
        db = ada.ada_calibrate(
            ZTest1D, g=g, db=db, nB=5, tile_batch_size=1, backend=backend
        )

    ip.testing.check_imprint_results(
        ip.Grid(db.get_results(), None).prune_inactive(), snapshot
    )


@pytest.mark.slow
def test_four_zones(duckdb, snapshot):
    four_zones_tester(duckdb, snapshot)


@pytest.mark.slow
def test_four_zones_ch(ch_db, snapshot):
    snapshot.set_test_name("test_four_zones")
    four_zones_tester(ch_db, snapshot)


@pytest.mark.slow
def test_four_zones_distributed(snapshot, ch_db):
    from confirm.adagrid.modal_backend import ModalBackend

    snapshot.set_test_name("test_four_zones")
    four_zones_tester(
        ch_db,
        snapshot,
        backend=ModalBackend(n_zones=4, coordinate_every=1, gpu=False),
    )


@pytest.mark.slow
def test_calibration_checkpointing():
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ada.db.DuckDBTiles.connect()
        _ = ada.ada_calibrate(ZTest1D, g=g, db=db, nB=3, init_K=4, n_steps=2)

        db_two2 = ada.ada_calibrate(ZTest1D, db=db, overrides=dict(n_steps=3))
        reports_two = db_two2.get_reports()

    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        db_once = ada.ada_calibrate(ZTest1D, g=g, nB=3, init_K=4, n_steps=3)
        reports_once = db_once.get_reports()

    assert len(reports_two) == len(reports_once) + 1
    df_reports_one = pd.DataFrame(reports_once)
    drop_cols = ["worker_id"] + [
        c for c in df_reports_one.columns if c.startswith("runtime")
    ]
    df_reports_one = df_reports_one.drop(drop_cols, axis=1)
    df_reports_two = (
        pd.DataFrame(reports_two)
        .drop(3, axis=0)
        .drop(drop_cols, axis=1)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(df_reports_two, df_reports_one)

    nondeterministic_cols = ["creator_id", "processor_id"]
    df_once = db_once.get_results().drop(nondeterministic_cols, axis=1)
    df_twice = db_two2.get_results().drop(nondeterministic_cols, axis=1)
    pd.testing.assert_frame_equal(df_twice, df_once)


def test_calibration_nonadagrid_using_adagrid():
    g = ip.cartesian_grid([-1], [1], n=[10], null_hypos=[ip.hypo("x < 0")])
    K = 2**13
    db = ada.ada_calibrate(
        ZTest1D,
        g=g,
        init_K=K,
        n_K_double=0,
        grid_target=0,
        bias_target=0,
        std_target=0,
        step_size=2**20,
        n_steps=1,
        tile_batch_size=1,
        prod=False,
    )
    results_df_nonada = ip.calibrate(ZTest1D, g=g, K=K, tile_batch_size=1)
    results_df_ada = db.get_results()
    pd.testing.assert_frame_equal(
        results_df_ada[results_df_nonada.columns], results_df_nonada
    )


def main():
    pass
    # import confirm.cloud.clickhouse as ch

    # pd.testing.assert_frame_equal(subset, snapshot(subset))

    # g = ip.cartesian_grid(theta_min=[-1], theta_max=[1],
    # null_hypos=[ip.hypo("x0 < 0")])
    # db = ch.Clickhouse.connect()
    # # db = ch.Clickhouse.connect(
    # #     host='localhost', port='8123', username='default', password='')
    # db = ada.ada_calibrate(
    #     ZTest1D, db=db, g=g, nB=5, tile_batch_size=1, n_iter=8
    # )
    # g = ip.cartesian_grid(
    #     theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
    # )
    # db = ada.ada_calibrate(ZTest1D, g=g, nB=2, prod=False,
    # tile_batch_size=1)


if __name__ == "__main__":
    main()
