import jax
import jax.numpy as jnp

from confirm.lewislib import batch


class LewisConcept:
    def __init__(self, key, max_nsims):
        # general setup that is independent of the tiles
        self.thresh = 20
        self.n_arms = 3
        self.n_arm_samples = 100
        self.unifs = jax.random.uniform(
            key, shape=(max_nsims, self.n_arm_samples, self.n_arms)
        )

        # 1. batch over simulations since we don't want to run huge numbers of
        #    simulation all at once and run out of memory. (grid point batching
        #    will happen outside the model)
        # 2. double vmap over both grid points and simulations since
        #    that is ideal for runtime and minimizing compilation and
        #    overhead.
        self._batched_sim = batch.batch(
            jax.vmap(jax.vmap(self._sim, in_axes=(0, None)), in_axes=(None, 0)),
            1000,  # the number of simulations to run in a single batch.
            (0, None),
        )

    def _sim(self, u, t):
        n = u.shape[0]
        stats = jnp.sum(u < jax.scipy.special.expit(t), axis=0) / n
        best_idx = jnp.argmax(stats[1:]) + 1
        best_stat = jnp.max(stats[1:])
        return best_stat, best_idx

    def sim_batch(self, nsims, theta, null_truth, detailed=False):
        # simulation-related info consistently comes before grid-pt related info.
        # so nsims is first here. and unifs_subset is first in the _batched_sim call.
        npts = theta.shape[0]
        test_stats, best_arms = self._batched_sim(self.unifs[:nsims], theta)
        false_test_stats = jnp.where(
            null_truth[jnp.arange(npts), best_arms - 1], test_stats, jnp.inf
        )
        if detailed:
            arm_idxs, arm_counts = jnp.unique(best_arms, return_counts=True)
            print(arm_idxs, arm_counts)
            return false_test_stats, arm_counts
        else:
            return false_test_stats


def sim_rej(model, nsims, theta, null_truth, lam, detailed=False):
    if hasattr(model, "sim_rej"):
        return model.sim_rej(nsims, theta, null_truth, lam, detailed)
    sim_res = model.sim_batch(nsims, theta, null_truth, detailed)
    if not detailed:
        sim_res = [sim_res]
    test_stats = sim_res[0]
    rej = jnp.sum(test_stats < lam, axis=-1)
    if detailed and len(sim_res) > 1:
        return rej, sim_res[1:]
    else:
        return rej


def sim_tune(model, nsims, theta, null_truth, alpha, order, detailed=False):
    if hasattr(model, "sim_tune"):
        return model.sim_tune(nsims, theta, null_truth, alpha, order, detailed)
    sim_res = model.sim_batch(nsims, theta, null_truth, detailed)
    if not detailed:
        sim_res = [sim_res]
    test_stats = sim_res[0]
    lamstar = _tune(nsims, test_stats, alpha)
    if detailed and len(sim_res) > 1:
        return lamstar, sim_res[1:]
    else:
        return lamstar


def _tune(nsims, stats, alpha):
    sorted_stats = jnp.sort(stats, axis=-1)
    cv_idx = jnp.maximum(
        jnp.floor((nsims + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0
    )
    return sorted_stats[..., cv_idx]


if __name__ == "__main__":
    N = 1000
    model = LewisConcept(jax.random.PRNGKey(0), N)
    stats, arm_counts = model.sim_batch(
        N, jnp.array([[0, 0, -0.2]]), jnp.array([[True, True]]), detailed=True
    )
    print(stats.shape, arm_counts.shape)
    print(arm_counts)

    rej = sim_rej(model, N, jnp.array([[0, 0, -0.2]]), jnp.array([[True, True]]), 0.5)
    print(rej)

    lamstar = sim_tune(
        model,
        N,
        jnp.array([[0, 0, -0.2]]),
        jnp.array([[True, True]]),
        0.05,
        jnp.arange(N),
    )
    print(lamstar)
