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
    # anywhere that we have a correct rejection, stick in 2 for the test stat so
    # that we can be sure to never include those simulations as false rejections.
    false_test_stats = jnp.where(null_truth[best_arms - 1], test_stats, 2.0)
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


def grouped_by_sim_size(lei_obj, f, max_grid_batch_size, n_out=1):
    n_arm_samples = int(lei_obj.unifs_shape()[0])

    def internal(sim_sizes, tile_args, unifs, unifs_order):
        unique_sizes = np.unique(sim_sizes)
        outs = [None] * n_out
        for size in unique_sizes:
            idx = sim_sizes == size
            start = time.time()
            grid_batch_size = min(int(1e9 / n_arm_samples / size), max_grid_batch_size)

            # batch over the tile args and not the sim args.
            in_axes = [None] + [0] * len(tile_args) + [None, None]

            f_batched = batch.batch_all_concat(f, grid_batch_size, in_axes=in_axes)
            res = f_batched(
                lei_obj, *[ta[idx] for ta in tile_args], unifs[:size], unifs_order
            )
            for i in range(n_out):
                if outs[i] is None:
                    outs[i] = np.empty(
                        (tile_args[0].shape[0], *res[i].shape[1:]), dtype=res[i].dtype
                    )
                outs[i][idx] = res[i]
            end = time.time()
            print(
                "running for size", size, "with", np.sum(idx), "tiles took", end - start
            )
        if n_out == 1:
            return outs[0]
        else:
            return outs

    return internal
