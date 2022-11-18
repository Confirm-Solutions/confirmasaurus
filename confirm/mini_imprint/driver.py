import copy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats

from . import grid


# TODO: generalize to other families
# TODO: ideally we'd have a single entrypoint function that does what
# `get_backward_bound` and `get_forward_bound` do??
def get_backward_bound(bound_module, family_params):
    scale = family_params.get("scale", 1.0)

    def backward_bound(alpha_target, theta0, vertices):
        v = vertices - theta0
        bwd_solver = bound_module.TileBackwardQCPSolver(scale=scale)
        q_opt = bwd_solver.solve(v, alpha_target)
        return bound_module.tilt_bound_bwd_tile(q_opt, scale, v, alpha_target)

    return jax.jit(jax.vmap(backward_bound, in_axes=(None, 0, 0)))


def get_forward_bound(bound_module, family_params):
    scale = family_params.get("scale", 1.0)

    def forward_bound(f0, theta0, vertices):
        fwd_solver = bound_module.TileForwardQCPSolver(scale=scale)
        vs = vertices - theta0
        q_opt = fwd_solver.solve(vs, f0)
        return bound_module.tilt_bound_fwd_tile(q_opt, scale, vs, f0)

    return jax.jit(jax.vmap(forward_bound))


def get_bound(model):
    family_params = model.family_params if hasattr(model, "family_params") else {}

    if model.family == "normal":
        import confirm.bound.normal as bound_module

    elif model.family == "binomial":
        raise Exception("not implemented")
    else:
        raise Exception("unknown family")

    return (
        get_forward_bound(bound_module, family_params),
        get_backward_bound(bound_module, family_params),
    )


def clopper_pearson(TI_sum, K, delta):
    TI_cp_bound = scipy.stats.beta.ppf(1 - delta, TI_sum + 1, K - TI_sum)
    # If typeI_sum == sim_sizes, scipy.stats outputs nan. Output 0 instead
    # because there is no way to go higher than 1.0
    return np.where(np.isnan(TI_cp_bound), 0, TI_cp_bound)


def calc_tuning_threshold(sorted_stats, sorted_order, alpha):
    K = sorted_stats.shape[0]
    cv_idx = jnp.maximum(jnp.floor((K + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0)
    # indexing a sorted array with sorted indices results in a sorted array!!
    return sorted_stats[sorted_order[cv_idx]]


class Driver:
    def __init__(self, model):
        self.model = model
        self.forward_boundv, self.backward_boundv = get_bound(model)

        self.tunev = jax.jit(
            jax.vmap(
                calc_tuning_threshold,
                in_axes=(0, None, 0),
            )
        )

    def _stats(self, K_df):
        K = K_df["K"].iloc[0]
        K_g = grid.Grid(K_df)
        theta = K_g.get_theta()
        # TODO: batching
        stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
        return stats

    def stats(self, df):
        return df.groupby("K", group_keys=False).apply(self._stats)

    def _validate(self, K_df, lam, delta):
        K = K_df["K"].iloc[0]
        K_g = grid.Grid(K_df)
        theta = K_g.get_theta()

        stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
        TI_sum = jnp.sum(stats < lam, axis=-1)
        TI_cp_bound = clopper_pearson(TI_sum, K, delta)
        theta, vertices = K_g.get_theta_and_vertices()
        TI_bound = self.forward_boundv(TI_cp_bound, theta, vertices)

        return K_df.assign(TI_sum=TI_sum, TI_cp_bound=TI_cp_bound, TI_bound=TI_bound)

    def validate(self, df, lam, *, delta=0.01):
        return df.groupby("K", group_keys=False).apply(
            lambda K_df: self._validate(K_df, lam, delta)
        )

    def _tune(self, K_df, alpha):
        K = K_df["K"].iloc[0]
        K_g = grid.Grid(K_df)

        theta, vertices = K_g.get_theta_and_vertices()
        alpha0 = self.backward_boundv(alpha, theta, vertices)

        stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
        sorted_stats = jnp.sort(stats, axis=-1)
        bootstrap_lams = self.tunev(sorted_stats, np.arange(K), alpha0)
        return pd.DataFrame(bootstrap_lams, columns=["lams"])

    def tune(self, df, alpha):
        return df.groupby("K", group_keys=False).apply(
            lambda K_df: self._tune(K_df, alpha)
        )


def _setup(modeltype, g, model_seed, K, model_kwargs):
    g = copy.deepcopy(g)
    if K is not None:
        g.df["K"] = K
    else:
        # If K is not specified we just use a default value that's a decent
        # guess.
        default_K = 2**14
        if "K" not in g.df.columns:
            g.df["K"] = default_K
        g.df.loc[g.df["K"] == 0, "K"] = default_K

    if model_kwargs is None:
        model_kwargs = {}
    model = modeltype(model_seed, g.df["K"].max(), **model_kwargs)
    return Driver(model), g


def validate(modeltype, g, lam, *, delta=0.01, model_seed=0, K=None, model_kwargs=None):
    driver, g = _setup(modeltype, g, model_seed, K, model_kwargs)
    rej_df = driver.validate(g.df, lam, delta=delta)
    return rej_df


def tune(modeltype, g, *, model_seed=0, alpha=0.025, K=None, model_kwargs=None):
    driver, g = _setup(modeltype, g, model_seed, K, model_kwargs)
    tune_df = driver.tune(g.df, alpha)
    return tune_df
