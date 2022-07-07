import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import scipy.stats
import scipy.special

import inla
import smalljax

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


def fast_berry(sig2, n_arms=4):
    sigma2_n = sig2.shape[0]
    arms = np.arange(n_arms)
    cov = np.full((sigma2_n, n_arms, n_arms), mu_sig2)
    cov[:, arms, arms] += sig2[:, None]
    neg_precQ = -np.linalg.inv(cov)

    neg_precQ_b = neg_precQ[:, 0, 1]
    neg_precQ_a = neg_precQ[:, 0, 0] - neg_precQ_b

    logprecQdet = 0.5 * jnp.log(jnp.linalg.det(-neg_precQ))
    log_prior = jnp.array(
        scipy.stats.invgamma.logpdf(sig2, sig2_alpha, scale=sig2_beta)
    )
    const = log_prior + logprecQdet
    solver = jax.vmap(jax.vmap(smalljax.gen(f"solve{n_arms}")))
    # invJI_logdet_vmap = jax.vmap(jax.vmap(inv_logdet_const_diag, in_axes=(0, 0, None)), in_axes=(0,None,None))
    dotJI_vmap = jax.vmap(jax.vmap(dotJI, in_axes=(0, 0, 0)), in_axes=(None,None,0))


    def log_joint(theta, p_pinned, data):
        y = data[..., 0]
        n = data[..., 1]
        theta_m0 = theta - mu_0
        theta_adj = theta + logit_p1
        exp_theta_adj = jnp.exp(theta_adj)
        quad = jnp.sum(
            theta_m0
            * (
                (theta_m0.sum(axis=-1) * neg_precQ_b[None])[..., None]
                + theta_m0 * neg_precQ_a[None, :, None]
            ),
            axis=-1,
        )
        return (
            0.5 * quad
            + jnp.sum(
                theta_adj * y[:, None] - n[:, None] * jnp.log(exp_theta_adj + 1),
                axis=-1,
            )
            + const
        )

    def step_hess(theta, _, data):
        y = data[..., 0]
        n = data[..., 1]
        theta_m0 = theta - mu_0
        exp_theta_adj = jnp.exp(theta + logit_p1)
        C = 1.0 / (exp_theta_adj + 1)
        nCeta = n[:, None] * C * exp_theta_adj
        grad = (
            # dotJI_vmap(neg_precQ_a, neg_precQ_b, theta_m0)
            + jnp.matmul(neg_precQ[None], theta_m0[:, :, :, None])[..., 0]
            + y[:, None]
            - nCeta
        )
        hess = neg_precQ[None] - ((nCeta * C)[:, :, None, :] * jnp.eye(n_arms))
        # hess_a = neg_precQ_a[None, :, None] - nCeta * C
        # hess_b = neg_precQ_b
        # ainv, binv, logdet = invJI_logdet_vmap(hess_a, hess_b, n_arms)
        step = -solver(hess, grad)
        return step, hess
        # return step, logdet

    return inla.FullLaplace(
        berry_model(n_arms),
        "sig2",
        np.zeros((n_arms, 2)),
        log_joint=log_joint,
        step_hess=step_hess,
        # logdet=lambda x: x
    )
    
def dotJI(a, b, v):
    return v.sum() * b + v * a

def inv_logdet_const_diag(a, b, d):
    def step(i, arg):
        for j in range(i):
            ai = a[i]
        logdet, ai, bi = arg
        step = bi * (bi / ai[i])
        return (logdet + jnp.log(jnp.abs(ai[i])), ai - step, bi - step)

    # Step 1) iteratively solve for the schur complement until we're left with a 2x2 in
    # the lower right corner.
    logdet, ai, bi = jax.lax.fori_loop(0, d - 2, step, (0.0, a, b))

    # Step 2) solve the remaining 2x2 system directly using a one step LU
    # decomposition
    a_final = ai - bi * (bi / ai)
    ainv = 1.0 / a_final
    binv = -ainv * bi / ai
    return ainv, binv, logdet
    