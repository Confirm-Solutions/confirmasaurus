import time

import jax
import jax.numpy as jnp
import numpy as np

from confirm.lewislib import batch


def sim(lei_obj, theta, unifs, unifs_order):
    p = jax.scipy.special.expit(theta)
    simulatev = jax.vmap(lei_obj.simulate, in_axes=(None, 0, None))
    test_stats, best_arms, _ = simulatev(p, unifs, unifs_order)
    return test_stats, best_arms


def tune(lei_obj, alpha, theta, null_truth, unifs, unifs_order):
    test_stats, best_arms = sim(lei_obj, theta, unifs, unifs_order)
    # anywhere that we have a correct rejection, stick in a large number for
    # the test stat so that we can be sure to never include those simulations
    # as false rejections.
    false_test_stats = jnp.where(null_truth[best_arms - 1], test_stats, 100.0)
    sim_cv = _tune(false_test_stats, unifs.shape[0], alpha)
    return (sim_cv,)


def _tune(test_stats, K, alpha):
    """
    Find the

    Args:
        test_stats: _description_
        K: _description_
        alpha: _description_

    Returns:
        _description_
    """
    sorted_stats = jnp.sort(test_stats, axis=-1)
    # cv_idx in the paper is 1-indexed, so we need to subtract 1 to be
    # 0-indexed
    # TODO: error if cv_idx = 0?
    # TODO: we could skip simulating when cv_idx = 0! free speedup!
    cv_idx = jnp.maximum(jnp.floor((K + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0)
    return sorted_stats[cv_idx]


def rej(lei_obj, cv, theta, null_truth, unifs, unifs_order):
    test_stats, best_arms = sim(lei_obj, theta, unifs, unifs_order)
    rej = test_stats < cv
    false_rej = rej * null_truth[best_arms - 1]
    return (np.sum(false_rej),)


simv = jax.jit(jax.vmap(sim, in_axes=(None, 0, None, None)), static_argnums=(0,))
tunev = jax.jit(
    jax.vmap(tune, in_axes=(None, 0, 0, 0, None, None)), static_argnums=(0,)
)
rejv = jax.jit(jax.vmap(rej, in_axes=(None, 0, 0, 0, None, None)), static_argnums=(0,))
# a second vmap to allow using multiple thresholds at once.
rejvv = jax.jit(
    jax.vmap(
        jax.vmap(rej, in_axes=(None, 0, None, None, None, None)),
        in_axes=(None, 0, 0, 0, None, None),
    ),
    static_argnums=(0,),
)


def get_sim_size_groups(sim_sizes):
    unique_sizes = np.unique(sim_sizes)
    for size in unique_sizes:
        idx = sim_sizes == size
        yield size, idx


def bootstrap_tune(
    lei_obj, alpha, theta, null_truth, unifs, bootstrap_idxs, unifs_order
):
    test_stats, best_arms = sim(lei_obj, theta, unifs, unifs_order)
    # anywhere that we have a correct rejection, stick in a large number for
    # the test stat so that we can be sure to never include those simulations
    # as false rejections.
    false_test_stats = jnp.where(null_truth[best_arms - 1], test_stats, 100.0)
    sim_cv = _tune(false_test_stats[bootstrap_idxs], unifs.shape[0], alpha)
    return (sim_cv,)


bootstrap_tunev = jax.jit(
    jax.vmap(
        jax.vmap(bootstrap_tune, in_axes=(None, None, None, None, None, 0, None)),
        in_axes=(None, 0, 0, 0, None, None, None),
    ),
    static_argnums=(0,),
)


def bootstrap_tune_runner(
    lei_obj, sim_sizes, alpha, theta, null_truth, unifs, bootstrap_idxs, unifs_order
):
    n_arm_samples = int(lei_obj.unifs_shape()[0])

    n_bootstraps = next(iter(bootstrap_idxs.values())).shape[0]
    out = np.empty((sim_sizes.shape[0], n_bootstraps), dtype=float)
    for size, idx in get_sim_size_groups(sim_sizes):
        # TODO: fix hardcoded 2**10
        grid_batch_size = min(int(1e9 / n_arm_samples / size), 2**10)
        # TODO: allow batch_all_concat to decide internally what batch size to
        # use.
        f_batched = batch.batch_all_concat(
            bootstrap_tunev,
            grid_batch_size,
            in_axes=(None, 0, 0, 0, None, None, None),
        )
        out[idx] = f_batched(
            lei_obj,
            alpha[idx],
            theta[idx],
            null_truth[idx],
            unifs[:size],
            bootstrap_idxs[size],
            unifs_order,
        )
    return out


# TODO: adapt to use the bootstrap_tune_runner design. it's better.
def grouped_by_sim_size(lei_obj, f, max_grid_batch_size, n_out=None):
    n_arm_samples = int(lei_obj.unifs_shape()[0])

    def internal(sim_sizes, tile_args, sim_args, *other_args):
        unique_sizes = np.unique(sim_sizes)
        outs = None
        _n_out = n_out
        for size in unique_sizes:
            idx = sim_sizes == size
            start = time.time()
            grid_batch_size = min(int(1e9 / n_arm_samples / size), max_grid_batch_size)

            # batch over the tile args and not the sim args.
            in_axes = [None] + [0] * len(tile_args) + [None] * len(sim_args) + [None]

            f_batched = batch.batch_all_concat(f, grid_batch_size, in_axes=in_axes)
            res = f_batched(
                lei_obj,
                *[ta[idx] for ta in tile_args],
                *[sa[:size] for sa in sim_args],
                *other_args
            )

            if _n_out is None:
                _n_out = len(res)
            if outs is None:
                outs = [None] * _n_out
            for i in range(_n_out):
                if outs[i] is None:
                    outs[i] = np.empty(
                        (tile_args[0].shape[0], *res[i].shape[1:]), dtype=res[i].dtype
                    )
                outs[i][idx] = res[i]
            end = time.time()
            print(
                "running for size", size, "with", np.sum(idx), "tiles took", end - start
            )
        if _n_out == 1:
            return outs[0]
        else:
            return outs

    return internal
