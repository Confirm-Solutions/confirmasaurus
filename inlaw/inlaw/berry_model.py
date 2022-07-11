import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import scipy.special
import scipy.stats

from inlaw import FullLaplace

mu_0 = -1.34
mu_sig2 = 100.0
sig2_alpha = 0.0005
sig2_beta = 0.000005
logit_p1 = scipy.special.logit(0.3)


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


def fast_berry(sig2, n_arms=4, dtype=np.float64, max_iter=30, tol=1e-3):
    sigma2_n = sig2.shape[0]
    arms = np.arange(n_arms)
    cov = np.full((sigma2_n, n_arms, n_arms), mu_sig2)
    cov[:, arms, arms] += sig2[:, None]
    neg_precQ = -np.linalg.inv(cov)
    logprecQdet = 0.5 * jnp.linalg.slogdet(neg_precQ)[1]
    log_prior = scipy.stats.invgamma.logpdf(sig2, sig2_alpha, scale=sig2_beta)

    neg_precQ_b = jnp.array(neg_precQ[:, 0, 1], dtype=dtype)
    neg_precQ_a = jnp.array(neg_precQ[:, 0, 0] - neg_precQ_b, dtype=dtype)
    const = jnp.array(log_prior + logprecQdet, dtype=dtype)
    dotJI_vmap = jax.vmap(jax.vmap(dotJI, in_axes=(0, 0, 0)), in_axes=(None, None, 0))
    solve_vmap = jax.vmap(jax.vmap(solve_basket), in_axes=(0, None, 0))

    def log_joint(theta, p_pinned, data):
        y = data[..., 0]
        n = data[..., 1]
        theta_m0 = theta - data.dtype.type(mu_0)
        theta_adj = theta + data.dtype.type(logit_p1)
        exp_theta_adj = jnp.exp(theta_adj)
        quad = jnp.sum(
            theta_m0 * (dotJI_vmap(neg_precQ_a, neg_precQ_b, theta_m0)),
            axis=-1,
        )
        return (
            data.dtype.type(0.5) * quad
            + jnp.sum(
                theta_adj * y[:, None] - n[:, None] * jnp.log(exp_theta_adj + 1),
                axis=-1,
            )
            + const
        )

    def step_hess(theta, _, data):
        y = data[..., 0]
        n = data[..., 1]
        theta_m0 = theta - dtype(mu_0)
        exp_theta_adj = jnp.exp(theta + dtype(logit_p1))
        C = 1 / (exp_theta_adj + 1)
        nCeta = n[:, None] * C * exp_theta_adj
        grad = dotJI_vmap(neg_precQ_a, neg_precQ_b, theta_m0) + y[:, None] - nCeta
        hess_a = neg_precQ_a[None, :, None] - nCeta * C
        hess_b = neg_precQ_b
        step, denom = solve_vmap(hess_a, hess_b, -grad)
        return step, (hess_a, denom)

    return FullLaplace(
        berry_model(n_arms),
        "sig2",
        np.zeros((n_arms, 2)),
        log_joint=log_joint,
        step_hess=step_hess,
        logdet=logdet,
        max_iter=max_iter,
        tol=tol,
    )


def dotJI(a, b, v):
    return v.sum() * b + v * a


def solve_basket(a, b, v):
    """
    solves the linear system:
    (bJ + diag(a - b)) x = v
    where J is the matrix of all ones, b is a scalar and a is a vector.
    """
    inv_a = 1 / a
    v_over_a = v * inv_a
    denom = 1 + (b * inv_a).sum()
    x = v_over_a - inv_a * (b * v_over_a.sum() / denom)
    return x, denom


def logdet(hess_info):
    hess_a, denom = hess_info
    return jnp.log(jnp.abs(denom)) + jnp.sum(jnp.log(jnp.abs(hess_a)), axis=-1)
