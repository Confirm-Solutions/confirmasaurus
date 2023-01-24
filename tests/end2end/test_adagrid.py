from unittest import mock

import numpy as np
import pandas as pd
import pytest
from jax.config import config

import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D

config.update("jax_enable_x64", True)


def check(db, snapshot, only_lams=False):
    lamss = db.worst_tile("lams")["lams"].iloc[0]
    np.testing.assert_allclose(lamss, snapshot(lamss))

    all_tiles_df = db.get_results().set_index("id")
    check_cols = ['step_id', "theta0", "radii0", "null_truth0"] + [
        c for c in all_tiles_df.columns if "lams" in c
    ]
    check_subset = (
        all_tiles_df[check_cols].sort_values(by=['step_id', "theta0"]).reset_index(drop=True)
    )
    compare = snapshot(check_subset)
    # SP = all_tiles_df.\
    #   sort_values(by=['theta0']).\
    #   reset_index(drop=True).\
    #   join(compare, rsuffix='_true')
    # SP.loc[(SP['twb_lams0'] - SP['twb_lams0_true']).nlargest().index]

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


@pytest.mark.slow
def test_adagrid(snapshot):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db = ada.ada_calibrate(ZTest1D, g=g, nB=5, tile_batch_size=1)
    check(db, snapshot)

    # Compare DuckDB against pandas
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        pd_db = ada.db.PandasTiles()
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        _, _, db2 = ada.ada_calibrate(ZTest1D, g=g, db=pd_db, nB=5, tile_batch_size=1)

    pd.testing.assert_frame_equal(
        db.get_results(),
        db2.get_results(),
        check_exact=True,
    )


@pytest.mark.slow
def test_adagrid_packetsize1(snapshot):
    snapshot.set_test_name("test_adagrid")
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, db = ada.ada_calibrate(
        ZTest1D, g=g, nB=5, tile_batch_size=1, packet_size=1
    )
    check(db, snapshot, only_lams=True)

@pytest.mark.slow
def test_adagrid_clickhouse(snapshot):
    snapshot.set_test_name("test_adagrid")
    import confirm.cloud.clickhouse as ch

    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        db = ch.Clickhouse.connect()
        iter, reports, db = ada.ada_calibrate(
            ZTest1D, g=g, db=db, nB=5, tile_batch_size=1
        )

    check(db, snapshot)


@pytest.mark.slow
def test_adagrid_clickhouse_distributed(snapshot):
    snapshot.set_test_name("test_adagrid")
    import confirm.cloud.clickhouse as ch
    import confirm.cloud.modal_util as modal_util
    import modal

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ch.Clickhouse.connect()
    iter, reports, db = ada.ada_calibrate(
        ZTest1D,
        g=g,
        db=db,
        nB=5,
        packet_size=1,
        n_iter=0,
        tile_batch_size=1,
    )

    job_id = db.job_id
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
        ada.ada_calibrate(ZTest1D, db=db, n_iter=100)

    with stub.run():
        list(worker.map(range(4)))

    check(db, snapshot, only_lams=True)


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
