import asyncio
import time
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


def cal_tester(snapshot, ignore_story=True, record_system=False, **kwargs):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        out_db = ada.ada_calibrate(
            ZTest1D, g=g, nB=5, tile_batch_size=1, record_system=record_system, **kwargs
        )

    ip.testing.check_imprint_results(
        ip.Grid(out_db.get_results(), None).prune_inactive(),
        snapshot,
        ignore_story=ignore_story,
    )


@pytest.mark.slow
def test_calibration(snapshot):
    cal_tester(snapshot, ignore_story=False)


@pytest.mark.slow
def test_calibration_packetsize1(snapshot):
    snapshot.set_test_name("test_calibration")
    cal_tester(snapshot, packet_size=1)


def distributed_tester(backend, ch_db, snapshot):
    import confirm.cloud.clickhouse as ch

    snapshot.set_test_name("test_calibration")
    db = ada.DuckDBTiles.connect()

    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        local_db = ada.ada_calibrate(
            ZTest1D,
            g=g,
            nB=5,
            record_system=False,
            clickhouse_service="TEST",
            db=db,
            job_name=ch_db.database,
            backend=backend,
        )

    ip.testing.check_imprint_results(
        ip.Grid(local_db.get_results(), None).prune_inactive(),
        snapshot,
        ignore_story=True,
    )

    restored_db = ada.DuckDBTiles.connect()
    i = 0
    # wait for all the asynchronous CH inserts to finish
    while i < 60:
        time.sleep(1)
        if (
            ch.query(db.ch_client, "select count(*) from results").result_set[0][0]
            == local_db.get_results().shape[0]
        ):
            break
        i += 1
    else:
        raise TimeoutError("Clickhouse results not restored")

    asyncio.run(ch.restore(db.ch_client, restored_db))
    snapshot.reset()
    ip.testing.check_imprint_results(
        ip.Grid(restored_db.get_results(), None).prune_inactive(),
        snapshot,
        ignore_story=True,
    )


@pytest.mark.slow
def test_distributed_modal(ch_db, snapshot):
    from confirm.cloud.modal_backend import ModalBackend

    backend = ModalBackend(gpu=False)
    distributed_tester(backend, ch_db, snapshot)


# @pytest.mark.skip
def test_distributed_coiled(ch_db, snapshot):
    from confirm.cloud.coiled_backend import CoiledBackend

    backend = CoiledBackend(restart_workers=True)
    distributed_tester(backend, ch_db, snapshot)


@pytest.mark.slow
def test_two_parallel_steps(snapshot):
    cal_tester(snapshot, n_parallel_steps=2, step_size=1, packet_size=1)


@pytest.mark.slow
def test_two_parallel_steps_distributed(snapshot):
    from confirm.cloud.modal_backend import ModalBackend

    snapshot.set_test_name("test_two_parallel_steps")
    cal_tester(
        snapshot,
        n_parallel_steps=2,
        step_size=1,
        packet_size=1,
        backend=ModalBackend(gpu=False),
    )


@pytest.mark.slow
def test_calibration_checkpointing():
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ada.db.DuckDBTiles.connect()
        _ = ada.ada_calibrate(
            ZTest1D, g=g, db=db, nB=3, init_K=4, n_steps=2, prod=False
        )

        db_two2 = ada.ada_calibrate(ZTest1D, db=db, overrides=dict(n_steps=3))
        reports_two = db_two2.get_reports()

    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        db_once = ada.ada_calibrate(ZTest1D, g=g, nB=3, init_K=4, n_steps=3, prod=False)
        reports_once = db_once.get_reports()

    assert len(reports_two) == len(reports_once)
    df_reports_one = pd.DataFrame(reports_once)
    drop_cols = [c for c in df_reports_one.columns if c.startswith("runtime")]
    df_reports_one = df_reports_one.drop(drop_cols, axis=1)
    df_reports_two = (
        pd.DataFrame(reports_two).drop(drop_cols, axis=1).reset_index(drop=True)
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
