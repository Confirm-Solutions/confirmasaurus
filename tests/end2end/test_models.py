import numpy as np
import pandas as pd
import scipy.stats

import confirm.mini_imprint as ip
from confirm.models.ztest import ZTest1D


def test_ztest(snapshot):
    g = ip.cartesian_grid([-1], [1], n=[10], null_hypos=[ip.hypo("x < 0")])
    # lam = -1.96 because we negated the statistics so we can do a less than
    # comparison.
    lam = -1.96
    K = 2**13
    rej_df = ip.validate(ZTest1D, g, lam, K=K)
    pd.testing.assert_frame_equal(rej_df, snapshot(rej_df))

    true_err = 1 - scipy.stats.norm.cdf(-g.get_theta()[:, 0] - lam)

    tie_est = rej_df["TI_sum"] / K
    tie_std = scipy.stats.binom.std(n=K, p=true_err) / K
    n_stds = (tie_est - true_err) / tie_std
    assert np.all(np.abs(n_stds) < 1.2)

    tune_df = ip.tune(ZTest1D, g)
    pd.testing.assert_frame_equal(tune_df, snapshot(tune_df))
