import copy
import jax
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


# def build_flat_log_likelihood(model, example_data):
#     # see https://num.pyro.ai/en/stable/handlers.html

#     trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(10))).get_trace(
#         example_data
#     )
#     slice_dict = dict()
#     i = 0
#     for k in trace:
#         if trace[k]["is_observed"]:
#             continue
#         shape = 1 if len(trace[k]["value"].shape) == 0 else trace[k]["value"].shape[0]
#         slice_dict[k] = (i, i + shape)
#         i += shape

#     def log_likelihood(params, data):
#         params_dict = dict()
#         for k, (i1, i2) in slice_dict:
#             params_dict[k] = params[i1:i2]
#         trace = handlers.trace(
#             handlers.substitute(handlers.seed(model, jax.random.PRNGKey(10)), params_dict)
#         ).get_trace(data)
#         return sum(
#             [jnp.sum(site["fn"].log_prob(site["value"])) for k, site in trace.items()]
#         )

#     return log_likelihood, slice_dict

def build_log_likelihood(model):
    def log_likelihood(params, data):
        trace = handlers.trace(
            handlers.substitute(handlers.seed(model, jax.random.PRNGKey(10)), params)
        ).get_trace(data)
        return sum(
            [jnp.sum(site["fn"].log_prob(site["value"])) for k, site in trace.items()]
        )

    return log_likelihood

def pin(ll_fnc, pinned_params):
    def wrapped(params, data):
        for k in pinned_params:
            if isinstance(pinned_params[k], np.ndarray):
                params[k] = jnp.where(
                    jnp.isnan(pinned_params[k]),
                    params[k],
                    pinned_params[k]
                )
            else:
                params[k] = pinned_params[k]
        return ll_fnc(params, data)
    return wrapped

class INLA:
    def __init__(self, log_joint, grad_hess, d):
        self.d = d
        self.log_joint = log_joint
        self.grad_hess = grad_hess
        self.solver = jax.vmap(jax.vmap(smalljax.gen(f"solve{d}")))

    @partial(jax.jit, static_argnums=(0, 3))
    def optimize_loop(self, data, sig2, tol):
        max_iter = 30

        def step(args):
            theta_max, hess, iters, go = args
            grad, hess = self.grad_hess(theta_max, sig2, data)
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
