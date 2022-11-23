import copy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats

from . import batching
from . import grid

# NOTE: See these GitHub issues/demo PRs for Model API discussion:
# https://github.com/Confirm-Solutions/confirmasaurus/pull/92
# https://github.com/Confirm-Solutions/confirmasaurus/pull/148


# TODO: Need to clean up the interface from driver to the bounds.
# TODO: Need to clean up the interface from driver to the bounds.
# TODO: Need to clean up the interface from driver to the bounds.
# TODO: Need to clean up the interface from driver to the bounds.
# - should the bound classes have staticmethods or should they be objects with
#   __init__?
# - can we pass a single vertex array as a substitute for the many vertex case?
def get_bound(family, family_params):
    if family == "normal":
        from confirm.bound.normal import NormalBound as bound_type
    elif family == "binomial":
        from confirm.bound.binomial import BinomialBound as bound_type
    else:
        raise Exception("unknown family")

    return (
        bound_type.get_forward_bound(family_params),
        bound_type.get_backward_bound(family_params),
    )


def clopper_pearson(tie_sum, K, delta):
    tie_cp_bound = scipy.stats.beta.ppf(1 - delta, tie_sum + 1, K - tie_sum)
    # If typeI_sum == sim_sizes, scipy.stats outputs nan. Output 0 instead
    # because there is no way to go higher than 1.0
    return np.where(np.isnan(tie_cp_bound), 0, tie_cp_bound)


def calc_tuning_threshold(sorted_stats, sorted_order, alpha):
    K = sorted_stats.shape[0]
    cv_idx = jnp.maximum(jnp.floor((K + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0)
    # indexing a sorted array with sorted indices results in a sorted array!!
    return sorted_stats[sorted_order[cv_idx]]


class Driver:
    def __init__(self, model, *, tile_batch_size):
        self.model = model
        self.tile_batch_size = tile_batch_size
        self.forward_boundv, self.backward_boundv = get_bound(
            model.family, model.family_params if hasattr(model, "family_params") else {}
        )

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

    def _batched_validate(self, K, theta, null_truth, lam):
        stats = self.model.sim_batch(0, K, theta, null_truth)
        return jnp.sum(stats < lam, axis=-1)

    def _validate(self, K_df, lam, delta):
        K = K_df["K"].iloc[0]
        K_g = grid.Grid(K_df)
        theta = K_g.get_theta()

        tie_sum = batching.batch(
            self._batched_validate,
            self.tile_batch_size,
            in_axes=(None, 0, 0, None),
        )(K, theta, K_g.get_null_truth(), lam)

        tie_cp_bound = clopper_pearson(tie_sum, K, delta)
        theta, vertices = K_g.get_theta_and_vertices()
        tie_bound = self.forward_boundv(tie_cp_bound, theta, vertices)

        return pd.DataFrame(
            dict(
                tie_sum=tie_sum,
                tie_est=tie_sum / K,
                tie_cp_bound=tie_cp_bound,
                tie_bound=tie_bound,
            )
        )

    def validate(self, df, lam, *, delta=0.01):
        # The execution trace of a driver:
        # entry point: (validate)
        #     for each K: (_validate)
        #         for each batch of tiles: (_batched_validate)
        #             simulate
        return df.groupby("K", group_keys=False).apply(
            lambda K_df: self._validate(K_df, lam, delta)
        )

    def _batched_tune(self, K, theta, vertices, null_truth, alpha):
        stats = self.model.sim_batch(0, K, theta, null_truth)
        sorted_stats = jnp.sort(stats, axis=-1)
        alpha0 = self.backward_boundv(alpha, theta, vertices)
        bootstrap_lams = self.tunev(sorted_stats, np.arange(K), alpha0)
        return bootstrap_lams

    def _tune(self, K_df, alpha):
        K = K_df["K"].iloc[0]
        K_g = grid.Grid(K_df)

        theta, vertices = K_g.get_theta_and_vertices()
        bootstrap_lams = batching.batch(
            self._batched_tune,
            self.tile_batch_size,
            in_axes=(None, 0, 0, 0, None),
        )(K, theta, vertices, K_g.get_null_truth(), alpha)
        return pd.DataFrame(bootstrap_lams, columns=["lams"])

    def tune(self, df, alpha):
        return df.groupby("K", group_keys=False).apply(
            lambda K_df: self._tune(K_df, alpha)
        )


def _setup(modeltype, g, model_seed, K, model_kwargs, tile_batch_size):
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
    return Driver(model, tile_batch_size=tile_batch_size), g


def validate(
    modeltype,
    g,
    lam,
    *,
    delta=0.01,
    model_seed=0,
    K=None,
    tile_batch_size=64,
    model_kwargs=None
):
    """
    Calculate the Type I Error bound.

    Args:
        modeltype: The model class.
        g: The grid.
        lam: The critical threshold in the rejection rule. Test statistics
             below this value will be rejected.
        delta: The bound will hold point-wise with probability 1 - delta.
               Defaults to 0.01.
        model_seed: The random seed. Defaults to 0.
        K: The number of simulations. If this is unspecified, it is assumed
           that the grid has a "K" column containing per-tile simulation counts.
           Defaults to None.
        tile_batch_size: The number of tiles to simulate in a single batch.
        model_kwargs: Keyword arguments passed to the model constructor.
                      Defaults to None.

    Returns:
        A dataframe with the following columns:
        - tie_sum: The number of test statistics below the critical threshold.
        - tie_est: The estimated Type I Error at the simulation points.
        - tie_cp_bound: The Clopper-Pearson bound on the Type I error at the
                        simulation point.
        - tie_bound: The bound on the Type I error over the whole tile.
    """
    driver, g = _setup(modeltype, g, model_seed, K, model_kwargs, tile_batch_size)
    rej_df = driver.validate(g.df, lam, delta=delta)
    return rej_df


def tune(
    modeltype,
    g,
    *,
    model_seed=0,
    alpha=0.025,
    K=None,
    tile_batch_size=64,
    model_kwargs=None
):
    """
    Tune the critical threshold for a given level of Type I Error control.

    Args:
        modeltype: The model class.
        g: The grid.
        model_seed: The random seed. Defaults to 0.
        alpha: The Type I Error control level. Defaults to 0.025.
        K: The number of simulations. If this is unspecified, it is assumed
           that the grid has a "K" column containing per-tile simulation counts.
           Defaults to None.
        tile_batch_size: The number of tiles to simulate in a single batch.
        model_kwargs: Keyword arguments passed to the model constructor.
           Defaults to None.

    Returns:
        _description_
    """
    driver, g = _setup(modeltype, g, model_seed, K, model_kwargs, tile_batch_size)
    tune_df = driver.tune(g.df, alpha)
    return tune_df