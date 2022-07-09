from dataclasses import dataclass
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers
import smalljax
from jax.config import config

# This line is critical for enabling 64-bit floats.
config.update("jax_enable_x64", True)

# TODO: laplace arm marginals
# TODO: conditional laplace/simplified laplace
# TODO: parameter checks and documentation!
# TODO: Test against MCMC using kullback-liebler.
# TODO: jax jit the batch_execute function
# TODO: i could pull the outer data.shape[0] vmap out and apply that to the
#       whole algorithm. THIS IS A GOOD IDEA.
# TODO: debug the problems with batching. why is it sometimes causing things to
#       randomly fail. add batching to pytest parameter grid
# TODO: why is convergence failing in 32 bit. it's not really. it just stalls
#       out when sig2 is very small.


class FullLaplace:
    def __init__(
        self,
        model,
        pin,
        data_example,
        max_iter=30,
        tol=1e-3,
        log_joint=None,
        step_hess=None,
        logdet=None,
    ):
        self.model = model
        self.pin = pin
        self.data_example = data_example
        self.spec = pin_to_spec(self.model, self.pin, self.data_example)
        self.d = self.spec.n_free
        self.log_joint = log_joint
        if self.log_joint is None:
            log_joint_single = build_log_likelihood(self.model)
            self.log_joint = jax.vmap(
                jax.vmap(
                    lambda x, p_pinned, data: log_joint_single(
                        self.spec.unravel_f(x), p_pinned, data
                    ),
                    in_axes=(0, 0, None),
                ),
                in_axes=(0, None, 0),
            )
            self.optimizer = build_optimizer(
                log_joint_single,
                self.spec,
                step_hess=step_hess,
                max_iter=max_iter,
                tol=tol,
            )
        else:
            self.optimizer = build_optimizer(
                None, None, step_hess=step_hess, max_iter=max_iter, tol=tol
            )
        self.calc_posterior = build_calc_posterior(
            self.log_joint, self.spec, logdet=logdet
        )
        self._jit_backend = jax.jit(self._backend)

    def __call__(
        self, p_pinned, data, x0=None, jit=True, should_batch=True, batch_size=2**12
    ):
        """
        batch:
            The batched execution mode runs chunks of a fixed number of
            inferences. Chunking and padding is a useful technique to avoid
            recompilation when calling FullLaplace multiple times with
            different data sizes. Each call to _call_backend results in a JAX
            JIT operation. These can be slow, especially in comparison to small
            amounts of data. Chunking is also useful for very large numbers of
            datasets because it relieves memory pressure and is often faster
            due to better use of caches

            The non-batched execution mode can sometimes be faster but incurs a
            substantial JIT cost when recompiling for each different input
            shape.
        """
        for k in self.spec.key_order:
            if k not in p_pinned:
                p_pinned[k] = None

        if x0 is None:
            pin_dim = pytree_shape0(p_pinned)
            x0 = jnp.zeros((data.shape[0], pin_dim, self.d), dtype=data.dtype)

        backend = self._jit_backend if jit else self._backend

        if should_batch:
            return batch_execute(
                backend,
                data.shape[0],
                batch_size,
                (p_pinned, False),
                (data, True),
                (x0, True),
            )
        else:
            return backend(p_pinned, data, x0)

    def _backend(self, p_pinned, data, x0):
        x_max, hess_info, iters = self.optimizer(x0, p_pinned, data)
        post = self.calc_posterior(x_max, hess_info, p_pinned, data)
        return post, x_max, hess_info, iters


def pytree_shape0(pyt):
    shape0 = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(lambda x: (x.shape[0] if x is not None else None), pyt)
    )[0]
    assert jnp.all(jnp.array(shape0) == shape0[0])
    return shape0[0]


def batch_execute(f, batch_dim, batch_size, *args):
    """
    An interesting jax problem is that many jit-ted JAX functions require
    re-compiling every time you change the array size of the inputs. There’s no
    general solution to this and if you’re running the same operation on a bunch of
    differently sized arrays, then the compilation time can become really annoying
    and expensive.

    One solution is to pad and batch your inputs. For example, we might decide
    to only run a function on chunks of 2^16 values. Then, any smaller set of
    values is padded with zeros out to 2^16. Any larger set of values is
    batched into a bunch of function calls. This has negative side effects
    (more complexity, potential performance consequences) but it succeeds in
    the goal of only ever compiling the function once.  It also often has the
    positive side effect of reducing memory usage.
    """
    n_batchs = int(np.ceil(batch_dim / batch_size))
    pad_N = batch_size * n_batchs

    def leftpad(arr):
        pad_spec = [[0, 0] for dim in arr.shape]
        rem = batch_dim % batch_size
        pad_spec[0][1] = batch_size - (batch_size if rem == 0 else rem)
        out = np.pad(arr, pad_spec)
        assert out.shape[0] == pad_N
        return out

    pad_args = [leftpad(a) if should_batch else a for a, should_batch in args]
    pad_out = None
    for i in range(n_batchs):
        start = i * batch_size
        end = (i + 1) * batch_size
        args_batch = [
            pad_args[i][start:end] if should_batch else pad_args[i]
            for i, (_, should_batch) in enumerate(args)
        ]
        out_batch = f(*args_batch)
        if pad_out is None:
            pad_out = []
            for arr in out_batch:
                shape = list(arr.shape)
                if len(shape) == 0:
                    shape = [pad_N]
                shape[0] = pad_N
                pad_out.append(np.empty_like(arr, shape=shape))
        for target, source in zip(pad_out, out_batch):
            target[start:end] = source
    return [arr[:batch_dim] for arr in pad_out]


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


def build_grad_hess(log_joint_single, param_spec):
    def grad_hess(x, p_pinned, data):
        # The inputs to grad_hess are pytrees but the output grad/hess are
        # flattened.
        p = param_spec.unravel_f(x)
        grad = jax.grad(log_joint_single)(p, p_pinned, data)
        hess = jax.hessian(log_joint_single)(p, p_pinned, data)

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

    return jax.vmap(jax.vmap(grad_hess, in_axes=(0, 0, None)), in_axes=(0, None, 0))


def build_optimizer(
    log_joint_single, param_spec, step_hess=None, max_iter=30, tol=1e-3
):
    if step_hess is None:
        if log_joint_single is None or param_spec is None:
            raise ValueError(
                "Either step_hess must be specified or both log_joint_single and "
                "param_spec must be specified."
            )
        grad_hess = build_grad_hess(log_joint_single, param_spec)
        solver = jax.vmap(jax.vmap(smalljax.gen(f"solve{param_spec.n_free}")))

        def step_hess(x, p_pinned, data):
            # Inputs and outputs are arrays, need to convert to pytrees internally.
            grad, hess = grad_hess(x, p_pinned, data)
            return -solver(hess, grad), hess

    def optimize_loop(x0, p_pinned, data):
        def step(args):
            x, hess_info, iters, go = args
            step, hess_info = step_hess(x, p_pinned, data)
            go = (jnp.max(jnp.sum(step**2)) > tol**2) & (iters < max_iter)
            return x + step, hess_info, iters + 1, go

        step0, hess_info0 = step_hess(x0, p_pinned, data)
        x, hess_info, iters, go = jax.lax.while_loop(
            lambda args: args[3], step, (x0 + step0, hess_info0, 1, True)
        )
        return x, hess_info, iters

    return optimize_loop


def build_calc_posterior(log_joint, param_spec, logdet=None):
    """
    The output will not be normalized!
    """
    if logdet is None:
        logdet = jax.vmap(jax.vmap(lambda x: smalljax.logdet(-x)))

    def calc_posterior(x_max, hess_info, p_pinned, data):
        lj = log_joint(x_max, p_pinned, data)
        log_post = lj - 0.5 * logdet(hess_info)
        log_post -= jnp.max(log_post)
        post = jnp.exp(log_post)
        return post

    return calc_posterior
