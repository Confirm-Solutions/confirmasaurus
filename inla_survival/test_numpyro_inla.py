import jax.numpy as jnp
import numpy as np
import scipy.stats
import numpyro
import numpyro.distributions as dist
from scipy.special import logit, expit

import inla

mu_0 = -1.34
mu_sig2 = 100
sig2_alpha = 0.0005
sig2_beta = 0.000005
logit_p1 = -1.0


def model(data):
    sig2 = numpyro.sample("sig2", dist.InverseGamma(sig2_alpha, sig2_beta))
    cov = jnp.full((3, 3), mu_sig2) + jnp.diag(jnp.repeat(sig2, 3))
    theta = numpyro.sample(
        "theta",
        dist.MultivariateNormal(mu_0, cov),
    )
    numpyro.sample(
        "y",
        dist.BinomialLogits(theta + logit_p1, total_count=data[..., 1]),
        obs=data[..., 0],
    )


def test_log_likelihood():
    params = dict(sig2=10.0, theta=np.array([0, 0, 0]))
    data = np.array([[6.0, 35], [5, 35], [4, 35]])
    ll = inla.build_log_likelihood(model)(params, data)

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
