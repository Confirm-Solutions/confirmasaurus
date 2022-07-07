import copy
from dataclasses import dataclass
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from jax.config import config

# This line is critical for enabling 64-bit floats.
config.update("jax_enable_x64", True)

# TODO: Test against MCMC.
# TODO: test multiple n_arms
# TODO: test automatic gradients and explicit gradients
# TODO: parameter checks!
# TODO: check dtypes input/output
# TODO: conditional laplace/simplified laplace
# TODO: integration factors
import smalljax

class ParamSpec:
    def __init__(self, param_example):
        # The ordering of entries in the concatenated grad/hess here will depend on
        # the order of entries in the param_example dictionary. Since Python 3.6,
        # dictionaries are insertion ordered so this depends on user choice of
        # parameter order. But, because we fix it once ahead of time, later
        # inconsistency will be just fine!
        self.param_example = param_example
        self.key_order = param_example.keys()
        self.not_nan = {k: ~jnp.isnan(param_example[k]) for k in self.key_order}
        self.dont_skip_idxs = {k: np.where(self.not_nan[k])[0] for k in self.key_order}
        self.n_params = sum([v.shape[0] for k, v in param_example.items()])
        self.n_pinned = sum([np.sum(np.isnan(v)) for k, v in param_example.items()])
        self.n_free = self.n_params - self.n_pinned

    def ravel_f(self, p, axis=-1):
        return jnp.concatenate(
            [p[k][self.not_nan[k]] for k in self.key_order if p[k] is not None],
            axis=axis,
        )

    def unravel_f(self, x):
        out = dict()
        i = 0
        for k in self.key_order:
            out_arr = jnp.full(self.param_example[k].shape[0], jnp.nan)
            end = i + self.dont_skip_idxs[k].shape[0]
            if end == i:
                out[k] = None
            else:
                out[k] = out_arr.at[self.dont_skip_idxs[k]].set(x[i:end])
            i = end
        return out

def pin_to_spec(model, pin, data_example):
    """
    Facilitates a clean user API.

    pin = "sig2"
    pin = ["sig2"]
    pin = [("sig2", 0)]
    pin = [("sig2", 0), ("theta", 1)]
    """
    seeded_model = handlers.seed(model, jax.random.PRNGKey(10))
    trace = handlers.trace(seeded_model).get_trace(data_example)
    d = {
        k: (v["value"].shape[0] if len(v["value"].shape) > 0 else 1)
        for k, v in trace.items()
        if not v["is_observed"]
    }

    param_example = {k: np.zeros(n) for k, n in d.items()}
    if not isinstance(pin, list):
        pin = [pin]
    for pin_entry in pin:
        if isinstance(pin_entry, str):
            param_example[pin_entry][:] = np.nan
        else:
            param_example[pin_entry[0]][pin_entry[1]] = np.nan
    return ParamSpec(param_example)

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


def build_grad_hess(log_joint, param_spec):
    def grad_hess(x, p_pinned, data):
        # The inputs to grad_hess are pytrees but the output grad/hess are
        # flattened.
        p = param_spec.unravel_f(x)
        grad = jax.grad(log_joint)(p, p_pinned, data)
        hess = jax.hessian(log_joint)(p, p_pinned, data)

        full_grad = param_spec.ravel_f(grad)
        full_hess = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        hess[k1][k2][param_spec.not_nan[k1]][:, param_spec.not_nan[k2]]
                        for k2 in param_spec.key_order
                        if hess[k1][k2] is not None
                    ],
                    axis=-1,
                )
                for k1 in param_spec.key_order
                if hess[k1] is not None
            ],
            axis=-2,
        )
        return full_grad, full_hess

    return jax.jit(
        jax.vmap(jax.vmap(grad_hess, in_axes=(0, 0, None)), in_axes=(0, None, 0))
    )


def build_step_hess(log_joint, param_spec):
    grad_hess = build_grad_hess(log_joint, param_spec)
    solver = jax.vmap(jax.vmap(smalljax.gen(f"solve{param_spec.n_free}")))

    def step_hess(x, p_pinned, data):
        # Inputs and outputs are arrays, need to convert to pytrees internally.
        grad, hess = grad_hess(x, p_pinned, data)
        return -solver(hess, grad), hess

    return step_hess


def build_optimizer(log_joint, param_spec, step_hess=None, max_iter=30, tol=1e-3):
    if step_hess is None:
        if log_joint is None or param_spec is None:
            raise ValueError(
                "Either step_hess must be specified or both log_joint and "
                "param_spec must be specified."
            )
        step_hess = build_step_hess(log_joint, param_spec)

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


def build_calc_posterior(log_joint, param_spec, logdet=None):
    """
    The output will not be normalized!
    """
    if logdet is None:
        logdet = jax.vmap(jax.vmap(lambda x: smalljax.logdet(-x)))

    log_joint_vmap = jax.vmap(
        jax.vmap(
            lambda x, p_pinned, data: log_joint(
                param_spec.unravel_f(x), p_pinned, data
            ),
            in_axes=(0, 0, None),
        ),
        in_axes=(0, None, 0),
    )

    def calc_posterior(x_max, hess_info, p_pinned, data):
        lj = log_joint_vmap(x_max, p_pinned, data)
        log_post = lj - 0.5 * logdet(hess_info)
        log_post -= jnp.max(log_post)
        post = jnp.exp(log_post)
        return post

    return jax.jit(calc_posterior)


def pytree_shape0(pyt):
    shape0 = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(lambda x: (x.shape[0] if x is not None else None), pyt)
    )[0]
    assert(jnp.all(jnp.array(shape0) == shape0[0]))
    return shape0[0]
    
class FullLaplace:
    def __init__(self, model, pin, data_example, max_iter=30, tol=1e-3):
        self.model = model
        self.pin = pin
        self.data_example = data_example
        self.spec = pin_to_spec(self.model, self.pin, self.data_example)
        self.N = self.spec.n_free
        self.log_joint = build_log_likelihood(self.model)
        self.optimizer = build_optimizer(self.log_joint, self.spec, max_iter=max_iter, tol=tol)
        self.calc_posterior = build_calc_posterior(self.log_joint, self.spec)
    
    def __call__(self, p_pinned, data, x0=None):
        if x0 is None:
            x0 = jnp.zeros((data.shape[0], pytree_shape0(p_pinned), self.N), dtype=data.dtype)
        x_max, hess_info, iters = self.optimizer(x0, p_pinned, data)
        post = self.calc_posterior(x_max, hess_info, p_pinned, data)
        return post, x_max, hess_info, iters
