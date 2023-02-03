from unittest import mock

import numpy as np
import pandas as pd
import pytest
from jax.config import config

import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D

config.update("jax_enable_x64", True)


def test_bootstrap_calibrate(snapshot):
    g = ip.cartesian_grid(
        theta_min=[-1], theta_max=[1], n=[10], null_hypos=[ip.hypo("x0 < 0")]
    )
    cal_df = ada.bootstrap.bootstrap_calibrate(ZTest1D, g=g, nB=5)
    twb_cols = [c for c in cal_df.columns if "twb_lams" in c]
    np.testing.assert_allclose(cal_df["twb_mean_lams"], cal_df[twb_cols].mean(axis=1))
    np.testing.assert_allclose(cal_df, snapshot(cal_df), rtol=1e-6)


def check(db, snapshot, only_lams=False):
    snapshot.set_test_name("test_calibration")
    lamss = db.worst_tile("lams")["lams"].iloc[0]
    np.testing.assert_allclose(lamss, snapshot(lamss))

    all_tiles_df = db.get_results().set_index("id")
    check_cols = ["step_id", "theta0", "radii0", "null_truth0"] + [
        c for c in all_tiles_df.columns if "lams" in c
    ]
    check_subset = (
        all_tiles_df[check_cols]
        .sort_values(by=["step_id", "theta0"])
        .reset_index(drop=True)
    )
    compare = snapshot(check_subset)

    # First check the calibration outputs. These are the most important values
    # to get correct.
    pd.testing.assert_frame_equal(check_subset, compare, check_dtype=False)
    if only_lams:
        return

    # Second, we check the remaining values. These are less important to be
    # precisely reproduced, but we still want to make sure they are
    # deterministic.
    pd.testing.assert_frame_equal(
        all_tiles_df,
        snapshot(all_tiles_df),
        check_like=True,
        check_index_type=False,
        check_dtype=False,
    )


def test_calibration_cheap(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, db = ada.ada_calibrate(
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
    results_df = db.get_results()
    subset = (
        results_df[
            ["step_id", "theta0"] + [c for c in results_df.columns if "lams" in c]
        ]
        .sort_values(by=["step_id", "theta0"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(subset, snapshot(subset))


@pytest.mark.slow
def test_calibration(snapshot):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db = ada.ada_calibrate(
            ZTest1D, g=g, nB=5, prod=False, tile_batch_size=1
        )
    check(db, snapshot)

    # Compare DuckDB against pandas
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        pd_db = ada.db.PandasTiles()
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        _, _, db2 = ada.ada_calibrate(
            ZTest1D, g=g, db=pd_db, nB=5, prod=False, tile_batch_size=1
        )

    pd.testing.assert_frame_equal(
        db.get_results(),
        db2.get_results(),
        check_exact=True,
    )


@pytest.mark.slow
def test_calibration_packetsize1(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, db = ada.ada_calibrate(
        ZTest1D, g=g, nB=5, tile_batch_size=1, packet_size=1
    )
    check(db, snapshot, only_lams=True)


@pytest.mark.slow
def test_calibration_clickhouse(snapshot, ch_db):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db = ada.ada_calibrate(
            ZTest1D, g=g, db=ch_db, nB=5, tile_batch_size=1
        )

    check(db, snapshot)


@pytest.mark.slow
def test_calibration_clickhouse_distributed(snapshot, ch_db):
    import confirm.cloud.modal_util as modal_util
    import modal

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, _ = ada.ada_calibrate(
        ZTest1D,
        g=g,
        db=ch_db,
        nB=5,
        packet_size=1,
        n_iter=0,
        tile_batch_size=1,
    )

    job_id = ch_db.job_id
    stub = modal.Stub("test_adagrid_clickhouse_distributed")

    @stub.function(
        image=modal_util.get_image(dependency_groups=["test", "cloud"]),
        retries=0,
        mounts=modal.create_package_mounts(["confirm", "imprint"]),
        secret=modal.Secret.from_name("kms-sops"),
        serialized=True,
    )
    def worker(i):
        import confirm.cloud.clickhouse as ch
        import confirm.adagrid as ada
        from imprint.models.ztest import ZTest1D
        from jax.config import config

        modal_util.decrypt_secrets()

        config.update("jax_enable_x64", True)
        db = ch.Clickhouse.connect(job_id=job_id)
        ada.ada_calibrate(ZTest1D, db=db, overrides=dict(n_iter=100))

    with stub.run():
        list(worker.map(range(4)))

    check(ch_db, snapshot, only_lams=True)


@pytest.mark.slow
def test_calibration_checkpointing():
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ada.db.DuckDBTiles.connect()
        iter_two1, reports_two1, db_two1 = ada.ada_calibrate(
            ZTest1D, g=g, db=db, nB=3, init_K=4, n_iter=2
        )

        iter_two2, reports_two2, db_two2 = ada.ada_calibrate(
            ZTest1D, db=db, overrides=dict(n_iter=1)
        )

    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], n=[3], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db_once = ada.ada_calibrate(
            ZTest1D, g=g, nB=3, init_K=4, n_iter=3
        )

    assert iter_two1 + iter_two2 == iter
    df_reports_one = pd.DataFrame(reports)
    drop_cols = ["worker_id", "worker_iter"] + [
        c for c in df_reports_one.columns if c.startswith("runtime")
    ]
    df_reports_one = df_reports_one.drop(drop_cols, axis=1)
    df_reports_two = pd.DataFrame(reports_two1 + reports_two2).drop(drop_cols, axis=1)
    pd.testing.assert_frame_equal(df_reports_two, df_reports_one)

    nondeterministic_cols = ["creator_id", "processor_id"]
    df_once = db_once.get_results().drop(nondeterministic_cols, axis=1)
    df_twice = db_two2.get_results().drop(nondeterministic_cols, axis=1)
    pd.testing.assert_frame_equal(df_twice, df_once)


def test_calibration_nonadagrid_using_adagrid(snapshot):
    g = ip.cartesian_grid([-1], [1], n=[10], null_hypos=[ip.hypo("x < 0")])
    K = 2**13
    iter, reports, db = ada.ada_calibrate(
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
    results_df_ada = results_df_ada[results_df_nonada.columns]
    pd.testing.assert_frame_equal(results_df_ada, results_df_nonada)
    pd.testing.assert_frame_equal(results_df_ada, snapshot(results_df_ada))


def main():
    pass
    # import confirm.cloud.clickhouse as ch

    # pd.testing.assert_frame_equal(subset, snapshot(subset))

    # g = ip.cartesian_grid(theta_min=[-1], theta_max=[1],
    # null_hypos=[ip.hypo("x0 < 0")])
    # db = ch.Clickhouse.connect()
    # # db = ch.Clickhouse.connect(
    # #     host='localhost', port='8123', username='default', password='')
    # iter, reports, db = ada.ada_calibrate(
    #     ZTest1D, db=db, g=g, nB=5, tile_batch_size=1, n_iter=8
    # )
    # g = ip.cartesian_grid(
    #     theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
    # )
    # iter, reports, db = ada.ada_calibrate(ZTest1D, g=g, nB=2, prod=False,
    # tile_batch_size=1)


if __name__ == "__main__":
    main()
