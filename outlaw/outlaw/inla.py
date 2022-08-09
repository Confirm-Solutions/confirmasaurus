"""
The INLA backend library.
See this github issue for a discussion of ways to mitigate JAX compilation times:
https://github.com/Confirm-Solutions/confirmasaurus/issues/22
"""
import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable
from typing import Dict
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from . import quad
from . import smalljax

# TODO: user interface
# TODO: deal with the scalar vs unit vector distinction in an elegant way:
#       currently sometimes unravel will produce a unit vector instead of scalar and
#       it can make writing a log joint function a bit trickier
# TODO: precise description of the operations, including call signatures
# TODO: it would be possible to specify a subset of the operation and then
#       derive the other operations from the specified subset.


@dataclass
class ParamSpec:
    param_example: np.ndarray
    key_order: List[str]
    not_nan: Dict[str, bool]
    dont_skip_idxs: Dict[str, np.ndarray]
    n_params: int
    n_pinned: int
    n_free: int

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


@dataclass
class Operations:
    """The operations needed to perform a variety of INLA calculations. The
    functions here are generated from a model or specified by the user.

    Operations that are suffixed with "v" are vectorized over the pinned
    variables dimension. This vectorization is done so that a custom model can
    perform optimization and pre-computations with respect to the pinned
    variables. For example, the custom basket trial model can pre-compute the
    the precision matrix and its determinant.
    """

    spec: ParamSpec
    log_jointv: Callable
    gradv: Callable
    hessv: Callable
    reduced_hessv: Callable
    logdetv: Callable
    step_hessv: Callable
    solve: Callable
    invert: Callable
    opt_tol: float = 1e-3
    max_iter: int = 30

    def config(self, max_iter=None, opt_tol=None):
        """Update the algorithmic configuration.

        Args:
            max_iter: The maximum number of optimizer iterations. If None
                (default), does not update.
            opt_tol: The optimizer tolerance. If None (default), does not update.

        Returns:
            A new Operations object with updated configuration.
        """
        out = copy.copy(self)
        out.opt_tol = opt_tol or self.opt_tol
        out.max_iter = max_iter or self.max_iter
        return out

    def find_mode(self, x0, p_pinned, data):
        def step(args):
            x, hess_info, iters, go = args
            step, hess_info = self.step_hessv(x, p_pinned, data)
            go = (jnp.max(jnp.sum(step**2)) > self.opt_tol**2) & (
                iters < self.max_iter
            )
            return x + step, hess_info, iters + 1, go

        step0, hess_info0 = self.step_hessv(x0, p_pinned, data)
        x, hess_info, iters, go = jax.lax.while_loop(
            lambda args: args[3], step, (x0 + step0, hess_info0, 1, True)
        )
        return x, hess_info, iters

    def integrate_cond_gaussian(self, x_max, hess_info, p_pinned, data):
        lj = self.log_jointv(x_max, p_pinned, data)
        log_post = lj - x_max.dtype.type(0.5) * self.logdetv(hess_info)
        return log_post

    def laplace_logpost(self, x0, p_pinned, data):
        x_max, hess_info, iters = self.find_mode(x0, p_pinned, data)
        logpost = self.integrate_cond_gaussian(x_max, hess_info, p_pinned, data)
        return logpost, x_max, hess_info, iters

    def cond_laplace_logpost(
        self, x_max, inv_hess_row, p_pinned, data, x, latent_idx, reduced=False
    ):
        cond_mu = mvn_conditional_meanv(x_max, inv_hess_row, x, latent_idx)
        if reduced:
            cond_hess = self.reduced_hessv(cond_mu, p_pinned, data, latent_idx)
        else:
            cond_mu = jnp.delete(cond_mu, latent_idx, axis=-1)
            cond_hess = self.hessv(cond_mu, p_pinned, data)
        return self.integrate_cond_gaussian(cond_mu, cond_hess, p_pinned, data)


def from_log_joint(log_joint, param_example):
    spec = ParamSpec(param_example)

    def split_log_joint(p, p_pinned, data):
        return log_joint(merge(p, p_pinned), data)

    def ravel_log_joint(x, p_pinned, data):
        return split_log_joint(spec.unravel_f(x), p_pinned, data)

    # The inputs to grad_hess are pytrees but the output grad/hess are
    # flattened.
    def grad(x, p_pinned, data):
        p = spec.unravel_f(x)
        grad = jax.grad(split_log_joint)(p, p_pinned, data)
        return spec.ravel_f(grad)

    def hess(x, p_pinned, data):
        p = spec.unravel_f(x)
        hess = jax.hessian(split_log_joint)(p, p_pinned, data)
        return jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        hess[k1][k2][spec.not_nan[k1]][:, spec.not_nan[k2]]
                        for k2 in spec.key_order
                        if hess[k1][k2] is not None
                    ],
                    axis=-1,
                )
                for k1 in spec.key_order
                if hess[k1] is not None
            ],
            axis=-2,
        )

    def reduced_hess(x, p_pinned, data, latent_idx):
        H = hess(x, p_pinned, data)
        return jnp.delete(H, latent_idx, axis=-1)

    solver = smalljax.gen(f"solve{spec.n_free}")

    def step_hess(x, p_pinned, data):
        # Inputs and outputs are arrays, need to convert to pytrees internally.
        g = grad(x, p_pinned, data)
        h = hess(x, p_pinned, data)
        return -solver(h, g), h

    return Operations(
        spec=spec,
        log_jointv=jax.vmap(ravel_log_joint, in_axes=(0, 0, None)),
        gradv=jax.vmap(grad, in_axes=(0, 0, None)),
        hessv=jax.vmap(hess, in_axes=(0, 0, None)),
        reduced_hessv=jax.vmap(reduced_hess, in_axes=(0, 0, None, None)),
        logdetv=jax.vmap(lambda x: smalljax.logdet(-x)),
        step_hessv=jax.vmap(step_hess, in_axes=(0, 0, None)),
        solve=solver,
        invert=smalljax.inv_recurse,
    )


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

    return jax.tree_util.tree_map(_merge, *pytrees, is_leaf=lambda x: x is None)


def mvn_conditional_mean(mu, cov, x, i):
    """Compute the conditional mean of a multivariate normal distribution
    conditioned on one of its variables

    Args:
        mu: The mean of the joint distribution
        cov: The covariance of the joint distribution
        x: The value of the variable conditioned on.
        i: The index of the variable conditioned on.

    Returns:
        The conditional mean
    """
    i_vec = jnp.eye(mu.shape[0], dtype=bool)[i]
    cov12 = jnp.where(i_vec, 0, cov)
    # When j == i, this is: x + 0 (because cov12 is 0 when j == i)
    # When j != i, this is mu + cov12 / ...
    return jnp.where(i_vec, x, mu) + cov12 / cov[i] * (x - mu[i])


mvn_conditional_meanv = jax.vmap(mvn_conditional_mean, in_axes=(0, 0, 0, None))


def pytree_shape0(pyt):
    shape0 = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(lambda x: (x.shape[0] if x is not None else None), pyt)
    )[0]
    assert jnp.all(jnp.array(shape0) == shape0[0])
    return shape0[0]


@partial(jax.jit, static_argnums=2)
def exp_and_normalize(log_d, wts, axis):
    log_d -= jnp.expand_dims(jnp.max(log_d, axis=axis), axis)
    d = jnp.exp(log_d)
    scaling_factor = jnp.sum(d * wts, axis=axis)
    d /= jnp.expand_dims(scaling_factor, axis)
    return d


def latent_grid(x_max, inv_hess_row, latent_idx, quad):
    """
    Returns a grid of quadrature points centered at the conditional modes of
    the densities under consideration. The width of the grid is determined by
    the standard deviation of the density as calculated from the provided
    hessian.

    Args:
        x_max: The mode of the density.
        hess: The hessian at the mode.
        latent_idx: The index of the latent variable to condition on.
        quad: The quadrature rule on an standardized domain. Note that if the
              domain of the input quadrature is [-1, 1], then the integral will
              compute over [-1 * std_dev, 1 * std_dev]. So, for the common case of
              integrating over 3 standard deviations, you will need to input a
              quadrature rule over the domain [-3, 3].

    Returns:
        The integration grid for the conditioned-on variable.
    """
    # TODO: what to do about negative values here? a negative value implies
    # that the hessian is not positive definite. this could be a saddle point
    # or local minimum caused by multi-modality. The goal here is to have some
    # measure of how wide to make our posterior density. Unfortunately, when
    # the density is far from gaussian (e.g. not positive definite!), the
    # hessian no longer provides a good measure of distributional width.
    # One solution in this situation would be to use an adaptive grid where we
    # extend left and right until the posterior density values are getting
    # small.
    sd = jnp.sqrt(jnp.abs(inv_hess_row[..., latent_idx]))
    pts = x_max[None, ..., latent_idx] + sd[None] * quad.pts[:, None, None]
    wts = sd[None] * quad.wts[:, None, None]
    return pts, wts


def gauss_hermite_grid(x_max, inv_hess_row, latent_idx, n=25):
    """
    See the docstring for `latent_grid`. This passes a Gauss-Hermite quadrature
    rule as the quadrature rule. Gauss-Hermite quadrature is nice for
    integrating from -inf to inf.
    """
    return latent_grid(x_max, inv_hess_row, latent_idx, quad.gauss_herm_rule(n))


def jensen_shannon_div(x, y, wts, axis):
    """
    Compute the Jensen-Shannon divergence between two distributions.

    Args:
        x: The first distribution.
        y: The second distribution.
        wts: The quadrature weights for integrating the relative entropy.
        axis: The integration/quadrature axis.

    Returns:
        The Jensen-Shannon divergence.
    """
    R = 0.5 * (x + y)

    def rel_entropy_integral(d):
        e = jnp.where(d == 0, 0, d * jnp.log(d / R))
        return jnp.sum(wts * e, axis=axis)

    a = rel_entropy_integral(x)
    b = rel_entropy_integral(y)
    return 0.5 * (a + b)
