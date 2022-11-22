import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats

import confirm.imprint as ip
import confirm.models.fisher_exact as fisher
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

    tie_est = rej_df["tie_sum"] / K
    tie_std = scipy.stats.binom.std(n=K, p=true_err) / K
    n_stds = (tie_est - true_err) / tie_std
    assert np.all(np.abs(n_stds) < 1.2)

    tune_df = ip.tune(ZTest1D, g)
    pd.testing.assert_frame_equal(tune_df, snapshot(tune_df))


def test_jax_hypergeom():
    np.testing.assert_allclose(
        fisher.hypergeom_logpmf(5, 20, 10, 10),
        scipy.stats.hypergeom.logpmf(5, 20, 10, 10),
    )
    np.testing.assert_allclose(
        fisher.hypergeom_logcdf(5, 20, 10, 10),
        scipy.stats.hypergeom.logcdf(5, 20, 10, 10),
    )
    np.testing.assert_allclose(
        jnp.exp(fisher.hypergeom_logcdf(5, 20, 10, 10)),
        scipy.stats.hypergeom.cdf(5, 20, 10, 10),
    )


def test_fisher_exact():
    model = fisher.FisherExact(0, 10, n_arm_samples=10)
    np.random.seed(0)
    theta = np.random.rand(5, 2)
    null_truth = np.ones((5, 1), dtype=bool)
    samples = model.samples
    p = jax.scipy.special.expit(theta)
    nsucc = jnp.sum(samples[None] < p[:, None, None], axis=2)
    tbl2by2 = np.concatenate(
        (nsucc[:, :, None, :], samples.shape[1] - nsucc[:, :, None, :]), axis=2
    )

    stats = np.array(
        [
            [
                1 - scipy.stats.fisher_exact(tbl2by2[i, j], alternative="greater")[1]
                for j in range(tbl2by2.shape[1])
            ]
            for i in range(tbl2by2.shape[0])
        ]
    )
    np.testing.assert_allclose(stats, model.sim_batch(0, 10, theta, null_truth))
