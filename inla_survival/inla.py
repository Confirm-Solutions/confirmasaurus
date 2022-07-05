import copy
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers

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


def build_log_likelihood(model):
    def log_likelihood(p, p_pinned, data):
        # Inputs are pytrees
        p_final = merge(p, p_pinned)
        seeded_model = handlers.seed(model, jax.random.PRNGKey(10))
        subs_model = handlers.substitute(seeded_model, p_final)
        trace = handlers.trace(subs_model).get_trace(data)
        return sum(
            [jnp.sum(site["fn"].log_prob(site["value"])) for k, site in trace.items()]
        )

    return log_likelihood

def build_ravel_fncs(param_ex):
    # The ordering of entries in the concatenated grad/hess here will depend on
    # the order of entries in the param_ex dictionary. Since Python 3.6,
    # dictionaries are insertion ordered so this depends on user choice of
    # parameter order. But, because we fix it once ahead of time, later
    # inconsistency will be just fine!
    key_order = param_ex.keys()
    def ravel_f(p, axis=-1):
        return jnp.concatenate(
            [
                p[k][~jnp.isnan(param_ex[k])]
                for k in key_order
                if param_ex[k] is not None
            ],
            axis=axis,
        )

    def unravel_f(x):
        out = dict()
        i = 0
        for k in key_order:
            if param_ex[k] is None:
                out[k] = None
                continue
            end = i + param_ex[k].shape[0]
            out_arr = jnp.full(param_ex[k].shape[0], jnp.nan)
            out[k] = out_arr.at[jnp.where(~jnp.isnan(param_ex[k]))[0]].set(x[i:end])
            i = end
        return out

    return ravel_f, unravel_f, key_order

def build_grad_hess(log_joint, param_ex, pin_ex):
    no_pinning = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: x is None, pin_ex)
    )
    if not no_pinning:
        isnan_pytree = [jnp.isnan(pin_ex[k]) for k in pin_ex.keys()]
        not_pinned = jnp.concatenate([v for v in isnan_pytree if not jnp.all(~v)])

    def grad_hess(p, p_pinned, data):
        # The inputs to grad_hess are pytrees but the output grad/hess are
        # flattened.
        grad = jax.grad(log_joint)(p, p_pinned, data)
        hess = jax.hessian(log_joint)(p, p_pinned, data)

        full_grad = jnp.concatenate(
            [grad[k] for k in p.keys() if grad[k] is not None], axis=-1
        )
        full_hess = jnp.concatenate(
            [
                jnp.concatenate(
                    [hess[k1][k2] for k2 in p.keys() if hess[k1][k2] is not None],
                    axis=-1,
                )
                for k1 in p.keys()
                if hess[k1] is not None
            ],
            axis=-2,
        )
        if no_pinning:
            return full_grad, full_hess
        else:
            return full_grad[not_pinned], full_hess[not_pinned][:, not_pinned]

    return jax.jit(
        jax.vmap(jax.vmap(grad_hess, in_axes=(0, 0, None)), in_axes=(0, None, 0))
    )


# TODO: how do I unflatten from the x array into the p dict
def build_step_hess(log_joint, pin_ex):
    grad_hess = build_grad_hess(log_joint, pin_ex)

    def step_hess(x, p_pinned, data):
        # Inputs and outputs are arrays, need to convert to pytrees internally.
        grad, hess = grad_hess(p, p_pinned, data)
        d = grad.shape[-1]
        solver = jax.vmap(jax.vmap(smalljax.gen(f"solve{d}")))
        return -solver(hess, grad)

    return step_hess


def build_optimizer(
    log_joint=None, pin_ex=None, step_hess=None, max_iter=30, tol=1e-3
):
    if step_hess is None:
        if log_joint is None or pin_ex is None:
            raise ValueError(
                "Either step_hess must be specified or both log_joint and "
                "pin_ex must be specified."
            )
        step_hess = build_grad_hess(log_joint, pin_ex)

    def optimize_loop(x0, p_pinned, data):
        def step(args):
            x, hess_info, iters, go = args
            step, hess_info = step_hess(x, p_pinned, data)
            go = jnp.any(jnp.sum(step**2) > tol**2) & (iters < max_iter)
            return x + step, hess_info, iters + 1, go

        step0, hess_info0 = step_hess(x0, p_pinned, data)
        x, hess_info, iters, go = jax.lax.while_loop(
            lambda args: args[3], step, (x0 + step0, hess_info0, 1, True)
        )
        return x, hess_info, iters

    return jax.jit(optimize_loop)


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
