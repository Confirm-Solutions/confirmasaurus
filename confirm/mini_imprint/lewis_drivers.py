"""
TODO:
TODO:
TODO:
TODO:
TODO:
Lots of duplication here with code that is in:
- binomial_tuning.py
- binomial.py
- execute.py

For now, I'm going to leave it as is, but as soon as we have the paper done, we
should clean this up.
"""
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np

from confirm.lewislib import batch


def simulate(lei_obj, theta, unifs, unifs_order):
    p = jax.scipy.special.expit(theta)
    simulatev = jax.vmap(lei_obj.simulate, in_axes=(None, 0, None))
    test_stats, best_arms, _ = simulatev(p, unifs, unifs_order)
    return test_stats, best_arms


def tune(lei_obj, alpha, theta, null_truth, unifs, unifs_order):
    test_stats, best_arms = simulate(lei_obj, theta, unifs, unifs_order)
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

    # NOTE: if cv_idx == 0, then the tuning problem is impossible. In this
    # situation, the caller is expected to deal with the issue by checking
    # alpha and K. Because this is inside a jax.jit function, we can't raise
    # an exception. An alternative would be to return a sentinel value, but for
    # now we'll just return sorted_stats[0] and hope that the caller handles
    # the problem.
    #
    # cv_idx in the paper is 1-indexed, so we need to subtract 1 to be
    # 0-indexed
    cv_idx = jnp.maximum(jnp.floor((K + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0)
    return sorted_stats[cv_idx]


def rej(lei_obj, cv, theta, null_truth, unifs, unifs_order):
    test_stats, best_arms = simulate(lei_obj, theta, unifs, unifs_order)
    rej = test_stats < cv
    false_rej = rej * null_truth[best_arms - 1]
    return (np.sum(false_rej),)


simv = jax.jit(jax.vmap(simulate, in_axes=(None, 0, None, None)), static_argnums=(0,))
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


def stat(lei_obj, theta, null_truth, unifs, unifs_order):
    test_stats, best_arms = simulate(lei_obj, theta, unifs, unifs_order)
    false_test_stats = jnp.where(null_truth[best_arms - 1], test_stats, 100.0)
    return false_test_stats


stat_jit = jax.jit(stat, static_argnums=(0,))


def memory_status(title):
    client = jax.lib.xla_bridge.get_backend()
    mem_usage = sum([b.nbytes for b in client.live_buffers()]) / 1e9
    print(f"{title} memory usage", mem_usage)
    print(f"{title} buffer sizes", [b.shape for b in client.live_buffers()])


def one_stat(lei_obj, theta, null_truth, K, unifs, unifs_order):
    # memory_status('one_stat')
    unifs_chunk = unifs[:K]
    out = batch.batch(stat_jit, int(1e4), in_axes=(None, None, None, 0, None),)(
        lei_obj,
        theta,
        null_truth,
        unifs_chunk,
        unifs_order,
    )
    del unifs_chunk
    return out


statv = jax.jit(jax.vmap(stat, in_axes=(None, 0, 0, None, None)), static_argnums=(0,))


def tune(sorted_stats, sorted_order, alpha):
    K = sorted_stats.shape[0]
    cv_idx = jnp.maximum(jnp.floor((K + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0)
    # indexing a sorted array with sorted indices results in a sorted array!!
    return sorted_stats[sorted_order[cv_idx]]


jit_tune = jax.jit(tune)
tunev = jax.jit(jax.vmap(jax.vmap(tune, in_axes=(None, 0, None)), in_axes=(0, None, 0)))


def bootstrap_tune_runner(
    lei_obj,
    sim_sizes,
    alpha,
    theta,
    null_truth,
    unifs,
    bootstrap_idxs,
    unifs_order,
    sim_batch_size=1024,
    grid_batch_size=64,
):
    n_bootstraps = next(iter(bootstrap_idxs.values())).shape[0]
    out = np.empty((sim_sizes.shape[0], n_bootstraps), dtype=float)
    for (size, idx, stats) in _stats_backend(
        lei_obj,
        sim_sizes,
        theta,
        null_truth,
        unifs,
        unifs_order,
        sim_batch_size,
        grid_batch_size,
    ):
        sorted_stats = np.sort(stats, axis=-1)
        res = tunev(sorted_stats, bootstrap_idxs[size], alpha[idx])
        out[idx] = res
    return out


def rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=1024,
    grid_batch_size=64,
):
    out = np.empty(sim_sizes.shape[0], dtype=int)
    for (_, idx, stats) in _stats_backend(
        lei_obj,
        sim_sizes,
        theta,
        null_truth,
        unifs,
        unifs_order,
        sim_batch_size,
        grid_batch_size,
    ):
        out[idx] = np.sum(stats < lam, axis=-1)
    return out


def _stats_backend(
    lei_obj,
    sim_sizes,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=1024,
    grid_batch_size=64,
):
    batched_statv = batch.batch(
        batch.batch(
            statv, sim_batch_size, in_axes=(None, None, None, 0, None), out_axes=(1,)
        ),
        grid_batch_size,
        in_axes=(None, 0, 0, None, None),
    )

    for size, idx in get_sim_size_groups(sim_sizes):
        print(
            f"simulating with K={size} and n_tiles={idx.sum()}"
            f" and batch_size=({grid_batch_size}, {sim_batch_size})"
        )
        start = time.time()
        unifs_chunk = unifs[:size]
        stats = batched_statv(
            lei_obj, theta[idx], null_truth[idx], unifs_chunk, unifs_order
        )
        del unifs_chunk
        gc.collect()
        print("simulation runtime", time.time() - start)

        yield (size, idx, stats)
