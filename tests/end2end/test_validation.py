from unittest import mock

import numpy as np
import pandas as pd
import pytest

import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D


def check(db, snapshot):
    snapshot.set_test_name("test_validation")

    # Leaving this here as an easy way to plot the results if debugging.
    # results = ip.Grid(db.get_results(), None).active()
    # import matplotlib.pyplot as plt
    # plt.plot(results.df["theta0"], results.df["tie_est"], 'ko')
    # plt.plot(results.df["theta0"], results.df["tie_cp_bound"], 'bo')
    # plt.plot(results.df["theta0"], results.df["tie_bound"], 'ro')
    # plt.show()

    max_tie = db.worst_tile("tie_bound")["tie_bound"].iloc[0]
    np.testing.assert_allclose(max_tie, snapshot(max_tie))

    all_tiles_df = db.get_results().set_index("id")
    check_cols = [
        "step_id",
        "theta0",
        "radii0",
        "null_truth0",
        "tie_sum",
        "tie_est",
        "tie_cp_bound",
        "tie_bound",
    ]
    check_subset = (
        all_tiles_df[check_cols]
        .sort_values(by=["step_id", "theta0"])
        .reset_index(drop=True)
    )
    compare = snapshot(check_subset)

    pd.testing.assert_frame_equal(check_subset, compare, check_dtype=False)


@pytest.mark.slow
def test_validation(snapshot):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db = ada.ada_validate(ZTest1D, g=g, lam=-1.96, tile_batch_size=1)
    check(db, snapshot)


@pytest.mark.slow
def test_validation_clickhouse(snapshot, ch_db):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db = ada.ada_validate(
            ZTest1D, g=g, db=ch_db, lam=-1.96, packet_size=1, tile_batch_size=1
        )

    check(db, snapshot)
