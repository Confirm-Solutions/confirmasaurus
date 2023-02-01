from unittest import mock

import numpy as np
import pandas as pd
import pytest

import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D


@pytest.mark.slow
def test_calibration(snapshot):
    with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        iter, reports, db = ada.ada_valibrate(
            ZTest1D, g=g, lam=1.96, nB=5, tile_batch_size=1
        )

    lamss = db.worst_tile("tie_bound")["lams"].iloc[0]
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
