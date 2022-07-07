import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import numpyro
import numpyro.distributions as dist
from scipy.special import logit, expit

import inla
import util

from jax.config import config

config.update("jax_enable_x64", True)

mu_0 = -1.34
mu_sig2 = 100
sig2_alpha = 0.0005
sig2_beta = 0.000005
logit_p1 = logit(0.3)


def berry_model(d):
    def model(data):
        sig2 = numpyro.sample("sig2", dist.InverseGamma(sig2_alpha, sig2_beta))
        cov = jnp.full((d, d), mu_sig2) + jnp.diag(jnp.repeat(sig2, d))
        theta = numpyro.sample(
            "theta",
            dist.MultivariateNormal(mu_0, cov),
        )
        numpyro.sample(
            "y",
            dist.BinomialLogits(theta + logit_p1, total_count=data[..., 1]),
            obs=data[..., 0],
        )

    return model


def test_log_likelihood():
    params = dict(sig2=10.0, theta=np.array([0, 0, 0]))
    data = np.array([[6.0, 35], [5, 35], [4, 35]])
    ll = inla.build_log_likelihood(berry_model(3))(
        params, dict(sig2=None, theta=None), data
    )

    invgamma_term = scipy.stats.invgamma.logpdf(
        params["sig2"], sig2_alpha, scale=sig2_beta
    )
    cov = jnp.full((3, 3), mu_sig2) + jnp.diag(jnp.repeat(params["sig2"], 3))
    normal_term = scipy.stats.multivariate_normal.logpdf(
        params["theta"], np.repeat(mu_0, 3), cov
    )
    binomial_term = scipy.stats.binom.logpmf(
        data[..., 0], data[..., 1], expit(params["theta"] + logit_p1)
    )
    np.testing.assert_allclose(
        ll, invgamma_term + normal_term + sum(binomial_term), rtol=1e-6
    )


def test_merge():
    a, b = dict(sig2=10.0, theta=jnp.array([3, np.nan])), dict(
        sig2=None, theta=jnp.array([np.nan, 2])
    )
    out = inla.merge(a, b)
    assert out["sig2"] == 10.0
    np.testing.assert_allclose(out["theta"], [3, 2])

def test_ravel_fncs():
    tt = np.arange(4, dtype=np.float64)
    tt[1] = np.nan
    ex = dict(sig2=np.array([np.nan]), theta=tt)
    spec = inla.ParamSpec(ex)
    r = spec.ravel_f(ex)
    np.testing.assert_allclose(r, [0,2,3])
    ur = spec.unravel_f(r)
    assert(ur['sig2'] is None)
    np.testing.assert_allclose(ur['theta'], ex['theta'])
        

def test_grad_hess():
    data = np.array([[7, 35], [6.0, 35], [5, 35], [4, 35]])
    spec = inla.ParamSpec(dict(
        sig2=np.array([10.0]),
        theta=np.array([0.0, 0.0, 0, 0]),
    ))
    ll_fnc = inla.build_log_likelihood(berry_model(4))
    grad_hess_vmap = inla.build_grad_hess(ll_fnc, spec)
    grad, hess = grad_hess_vmap(
        np.concatenate((np.full((1, 1, 1), 10.0), np.zeros((1, 1, 4))), axis=2),
        dict(sig2=None, theta=None),
        data[None],
    )
    full_grad = np.array([-0.25124812, -3.5032682, -4.5032682, -5.5032682, -6.5032682])
    hess01 = np.array([
        [2.5007863e-02, 7.9709353e-06, 7.9717356e-06, 7.9713354e-06, 7.9718011e-06],
        [7.9710153e-06, -7.4256096e00, 2.4390254e-02, 2.4390249e-02, 2.4390247e-02],
    ])
    np.testing.assert_allclose(grad[0, 0], full_grad, rtol=1e-4)
    np.testing.assert_allclose(hess[0, 0, :2], hess01, rtol=1e-4)

    spec2 = inla.ParamSpec(dict(sig2=jnp.array([jnp.nan]), theta=jnp.array([0, np.nan, 0, 0])))
    grad_hess_vmap = inla.build_grad_hess(ll_fnc, spec2)
    theta_fixed = jnp.array([np.nan, 0.0, np.nan, np.nan])
    grad, hess = grad_hess_vmap(
        np.zeros((2, 1, 4)),
        dict(sig2=np.array([[10.0]]), theta=theta_fixed[None]),
        np.tile(data[None], (2, 1, 1)),
    )
    np.testing.assert_allclose(grad[0, 0], full_grad[[1,3,4]], rtol=1e-4)
    np.testing.assert_allclose(hess[0, 0, 0], hess01[1,[1,3,4]], rtol=1e-4)


def test_optimize_posterior():
    N = 10
    n_i = np.tile(np.array([20, 20, 35, 35]), (N, 1))
    y_i = np.tile(np.array([0, 1, 9, 10]), (N, 1))
    data = np.stack((y_i, n_i), axis=-1).astype(np.float64)
    ll_fnc = inla.build_log_likelihood(berry_model(4))

    sig2_rule = util.log_gauss_rule(15, 1e-6, 1e3)
    x0 = np.zeros((N, sig2_rule.pts.shape[0], 4))
    spec = inla.ParamSpec(dict(sig2=np.array([np.nan]), theta=np.zeros(4)))
    optimizer = inla.build_optimizer(ll_fnc, spec)
    p_pinned = dict(sig2=sig2_rule.pts, theta=None)
    x_max, hess, iters = optimizer(x0, p_pinned, data)
    np.testing.assert_allclose(
        x_max[0, 12],
        np.array([-6.04682818, -2.09586893, -0.21474981, -0.07019088]),
        atol=1e-3,
    )
    post = inla.build_calc_posterior(ll_fnc, spec)(x_max, hess, p_pinned, data)
    post /= np.sum(post * sig2_rule.wts, axis=1)[:, None]
    # post = infer.posterior(theta_max, hess, sig2_rule.pts, sig2_rule.wts, data)
    correct = np.array(
        [
            1.25954474e02,
            4.52520893e02,
            8.66625278e02,
            5.08333300e02,
            1.30365045e02,
            2.20403048e01,
            3.15183578e00,
            5.50967224e-01,
            2.68365061e-01,
            1.23585852e-01,
            1.13330444e-02,
            5.94800210e-04,
            4.01075571e-05,
            4.92782335e-06,
            1.41605356e-06,
        ]
    )
    np.testing.assert_allclose(post[0], correct, rtol=1e-3)
