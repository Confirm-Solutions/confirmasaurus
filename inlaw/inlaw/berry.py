import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import scipy.special
import scipy.stats

import inlaw.inla as inla

mu_0 = -1.34
mu_sig2 = 100.0
sig2_alpha = 0.0005
sig2_beta = 0.000005
logit_p1 = scipy.special.logit(0.3)


def figure2_data(N=10):
    n_i = np.tile(np.array([20, 20, 35, 35]), (N, 1))
    y_i = np.tile(np.array([0, 1, 9, 10]), (N, 1))
    return np.stack((y_i, n_i), axis=-1).astype(np.float64)


def model(d):
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


def log_joint(d):
    def ll(params, data):
        sig2 = params["sig2"]
        cov = jnp.full((d, d), mu_sig2) + jnp.diag(jnp.repeat(sig2, d))
        return (
            jnp.sum(dist.InverseGamma(sig2_alpha, sig2_beta).log_prob(sig2))
            + dist.MultivariateNormal(mu_0, cov).log_prob(params["theta"])
            + jnp.sum(
                dist.BinomialLogits(
                    params["theta"] + logit_p1, total_count=data[..., 1]
                ).log_prob(data[..., 0])
            )
        )

    return ll


def optimized(sig2, n_arms=4, dtype=np.float64):
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
    dotJI_vmap = jax.vmap(dotJI)
    solve_vmap = jax.vmap(solve_basket)

    def log_joint(theta, _, data):
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
                theta_adj * y[None] - n[None] * jnp.log(exp_theta_adj + 1),
                axis=-1,
            )
            + const
        )

    def grad_hess(theta, _, data):
        y = data[..., 0]
        n = data[..., 1]
        theta_m0 = theta - dtype(mu_0)
        exp_theta_adj = jnp.exp(theta + dtype(logit_p1))
        C = 1 / (exp_theta_adj + 1)
        nCeta = n[None] * C * exp_theta_adj
        grad = dotJI_vmap(neg_precQ_a, neg_precQ_b, theta_m0) + y[None] - nCeta
        hess_a = neg_precQ_a[:, None] - nCeta * C
        hess_b = neg_precQ_b
        return grad, (hess_a, hess_b)

    def grad(theta, _, data):
        return grad_hess(theta, _, data)[0]

    def hess(theta, _, data):
        return grad_hess(theta, _, data)[1]

    def reduced_hess(theta, _, data, drop_idx):
        H = hess(theta, _, data)
        return (jnp.delete(H[0], drop_idx, axis=-1), H[1])

    def step_hess(theta, _, data):
        grad, (hess_a, hess_b) = grad_hess(theta, _, data)
        step, denom = solve_vmap(hess_a, hess_b, -grad)
        return step, (hess_a, denom)

    return inla.Operations(
        spec=inla.ParamSpec(dict(sig2=np.array([np.nan]), theta=np.full(n_arms, 0))),
        log_jointv=log_joint,
        gradv=grad,
        hessv=hess,
        reduced_hessv=reduced_hess,
        logdetv=logdet_basket,
        step_hessv=step_hess,
        solve=lambda H, v: solve_basket(*H, v),
        invert=lambda H: inv_basket(*H),
    )


def dotJI(a, b, v):
    return v.sum() * b + v * a


def inv_basket(a, b):
    """Inverts the matrix arising from a bayesian basket trial:
    (aI + bJ) where b is a scalar and a is a vector.

    Args:
        a: vector of length n
        b: scalar

    Returns:
        The inverse of (aI + bJ)
    """
    inv_a = 1 / a
    denom = 1 + b * inv_a.sum()
    return jnp.diag(inv_a) - jnp.outer(inv_a, inv_a) * b / denom


def solve_basket(a, b, v):
    """Solves the linear systems arising from a bayesian basket trial:
    (aI + bJ) x = v
    where J is the matrix of all ones, b is a scalar and a is a vector.

    Args:
        a: vector of length n
        b: scalar
        v: vector of length n

    Returns:
        x: vector of length n
        denom: scalar (1 / (1 + b * (1 / a).sum())), this is useful for later
            computing the log determinant of the hessian.
    """
    inv_a = 1 / a
    v_over_a = v * inv_a
    denom = 1 + b * inv_a.sum()
    x = v_over_a - inv_a * (b * v_over_a.sum() / denom)
    return x, denom


def logdet_basket(hess_info):
    """The log determinant of a Basket trial hessian using the Matrix
    Determinant Lemma:
    https://en.wikipedia.org/wiki/Matrix_determinant_lemma

    Args:
        hess_info: A tuple (a, denom) where (aI + bJ) is the hessian. a is a vector
            and b is a scalar and denom = 1 / (1 + b * (1.0 / a).sum()).

    Returns:
        The log of the determinant of the hessian.
    """
    hess_a, denom = hess_info
    return jnp.log(jnp.abs(denom)) + jnp.sum(jnp.log(jnp.abs(hess_a)), axis=-1)


def build_dirty_bayes(sig2, n_arms=4, dtype=np.float64):
    """Partial implementation of Dirty Bayes.

    Args:
        sig2: The sigma2 integration grid.
        n_arms: Number of arms. Defaults to 4.
        dtype: Defaults to np.float64.

    Returns:
        A function implementing Dirty Bayes that accepts a data array and
        returns the mean and variance of the posterior.
    """
    sigma2_n = sig2.shape[0]
    arms = np.arange(n_arms)
    cov = np.full((sigma2_n, n_arms, n_arms), mu_sig2)
    cov[:, arms, arms] += sig2[:, None]
    prec = np.linalg.inv(cov)
    # logdet_prec = 0.5 * jnp.linalg.slogdet(prec)[1]
    # log_prior = scipy.stats.invgamma.logpdf(sig2, sig2_alpha, scale=sig2_beta)

    prec_b = jnp.array(prec[:, 0, 1], dtype=dtype)
    prec_a = jnp.array(prec[:, 0, 0] - prec_b, dtype=dtype)
    prec_mu_0 = jnp.array((prec * mu_0).sum(axis=-1), dtype=dtype)

    inv_basket_vmap = jax.vmap(inv_basket, in_axes=(0, 0))
    solve_basket_vmap = jax.vmap(solve_basket, in_axes=(0, 0, 0))

    def dirty_bayes(data):
        y = data[..., 0]
        n = data[..., 1]
        phat = y / n
        thetahat = jax.scipy.special.logit(phat) - logit_p1
        sample_I = n * phat * (1 - phat)
        prec_post_a = prec_a[:, None] + sample_I[None]
        sigma_post = inv_basket_vmap(prec_post_a, prec_b)
        mu_post = solve_basket_vmap(
            prec_post_a, prec_b, sample_I * thetahat + prec_mu_0
        )[0]
        return mu_post, jnp.diagonal(sigma_post, axis1=1, axis2=2)

    return dirty_bayes
