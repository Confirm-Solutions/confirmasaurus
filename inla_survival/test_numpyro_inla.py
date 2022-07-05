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
    ll = inla.build_log_likelihood(berry_model(3))(params, data)

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
    a, b = dict(sig2 = 10.0, theta = jnp.array([3, np.nan])), dict(sig2 = None, theta=jnp.array([np.nan, 2]))
    out = inla.merge(a, b)
    assert(out['sig2'] == 10.0)
    np.testing.assert_allclose(out['theta'], [3, 2])

def test_optimize_posterior():
    N = 10
    n_i = np.tile(np.array([20, 20, 35, 35]), (N, 1))
    y_i = np.tile(np.array([0, 1, 9, 10]), (N, 1))
    data = np.stack((y_i, n_i), axis=-1).astype(np.float64)
    ll_fnc = inla.build_raw_log_likelihood(berry_model(4))

    # def conditional(theta, sig2, data):
    #     params = dict(theta=theta, sig2=sig2)
    #     return ll_fnc(params, dict(sig2=None, theta=None), data)


    # def grad_hess(theta, sig2, data):
    #     grad = jax.grad(conditional)(theta, sig2, data)
    #     hess = jax.hessian(conditional)(theta, sig2, data)
    #     return grad, hess

    # grad_hess_vmap = jax.jit(
    #     jax.vmap(jax.vmap(grad_hess, in_axes=(0, 0, None)), in_axes=(0, None, 0))
    # )
    # conditional_vmap = jax.jit(
    #     jax.vmap(jax.vmap(conditional, in_axes=(0, 0, None)), in_axes=(0, None, 0))
    # )
    infer = inla.INLA(ll_fnc, 4)
    sig2_rule = util.log_gauss_rule(15, 1e-6, 1e3)
    theta_max, hess, iters = infer.optimize_loop(data, sig2_rule.pts, 1e-3)
    post = infer.posterior(theta_max, hess, sig2_rule.pts, sig2_rule.wts, data)
    np.testing.assert_allclose(
        theta_max[0, 12],
        np.array([-6.04682818, -2.09586893, -0.21474981, -0.07019088]),
        atol=1e-3,
    )
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