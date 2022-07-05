import copy
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers

# TODO: Tests for the inversion routines.
# TODO: Test against MCMC.
# TODO: test multiple n_arms
# TODO: test automatic gradients and explicit gradients
import smalljax


def merge(*pytrees):
    def _merge(*args):
        combine = None
        for arg in args:
            if arg is not None:
                if combine is not None:
                    overwrite = ~jnp.isnan(arg)
                    combine = jnp.where(overwrite, arg, combine)
                elif isinstance(arg, jnp.ndarray):
                    combine = arg
                else:
                    return arg
        return combine

    return jax.tree_map(_merge, *pytrees, is_leaf=lambda x: x is None)


def build_raw_log_likelihood(model, example_p):
    jax.tree_util.tree
    ravel_p, unravel_fnc = jax.flatten_util.ravel_pytree(example_p)
    ravel_fnc = lambda p: jax.flatten_util.ravel_pytree(p)[0]
    which_nan = jnp.isnan(ravel_p)
    mult = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)).astype(int), example_p)

    def log_likelihood(p, p_pinned, data):
        param_array = jnp.where(~jnp.array([True, False, False, False, False]), p, p_pinned)
        seeded_model = handlers.seed(model, jax.random.PRNGKey(10))
        subs_model = handlers.substitute(seeded_model, unravel_fnc(param_array))
        trace = handlers.trace(subs_model).get_trace(data)
        return sum(
            [jnp.sum(site["fn"].log_prob(site["value"])) for k, site in trace.items()]
        )

    return log_likelihood, ravel_fnc, unravel_fnc


def build_grad_hess(ll_fnc):
    def grad_hess(p, p_pinned, data):
        grad = jax.grad(ll_fnc)(p, p_pinned, data)
        hess = jax.hessian(ll_fnc)(p, p_pinned, data)
        return grad, hess
    return grad_hess

    # return jax.jit(
    #     jax.vmap(jax.vmap(grad_hess, in_axes=(0, 0, None)), in_axes=(0, None, 0))
    # )


class INLA:
    def __init__(self, log_joint, d, grad_hess=None):
        self.d = d
        self.log_joint = log_joint
        self.grad_hess = grad_hess
        if self.grad_hess is None:
            self.grad_hess = build_grad_hess(self.log_joint)
        self.solver = jax.vmap(jax.vmap(smalljax.gen(f"solve{d}")))

    @partial(jax.jit, static_argnums=(0, 3))
    def optimize_loop(self, data, sig2, tol):
        max_iter = 30

        def step(args):
            theta_max, hess, iters, go = args
            p1 = dict(theta=theta_max, sig2=None)
            p2 = dict(theta=None, sig2=sig2)
            grad, hess = self.grad_hess(p1, p2, data)
            step = -self.solver(hess, grad)
            go = jnp.any(jnp.sum(step**2) > tol**2) & (iters < max_iter)
            return theta_max + step, hess, iters + 1, go

        n_arms = data.shape[1]
        theta_max0 = jnp.zeros((data.shape[0], sig2.shape[0], n_arms))
        hess0 = jnp.zeros((data.shape[0], sig2.shape[0], n_arms, n_arms))
        init_args = (theta_max0, hess0, 0, True)

        out = jax.lax.while_loop(lambda args: args[3], step, init_args)
        theta_max, hess, iters, go = out
        return theta_max, hess, iters

    @partial(jax.jit, static_argnums=(0,))
    def posterior(self, theta_max, hess, sig2, wts, data):
        lj = self.log_joint(theta_max, sig2, data)
        logdet = jax.vmap(jax.vmap(smalljax.logdet))(-hess)
        log_post = lj - 0.5 * logdet
        log_post -= jnp.max(log_post)
        post = jnp.exp(log_post)
        return post / jnp.sum(post * wts, axis=1)[:, None]
