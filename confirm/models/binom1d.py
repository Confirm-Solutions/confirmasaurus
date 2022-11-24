import jax
import jax.numpy as jnp


@jax.jit
def _sim(samples, theta, null_truth):
    p = jax.scipy.special.expit(theta)
    stats = jnp.sum(samples[None, :] < p[:, None], axis=2) / samples.shape[1]
    return jnp.where(
        null_truth[:, None, 0],
        1 - stats,
        jnp.inf,
    )


def unifs(seed, *, shape, dtype):
    return jax.random.uniform(jax.random.PRNGKey(seed), shape=shape, dtype=dtype)


class Binom1D:
    def __init__(self, cache, seed, max_K, *, n):
        self.family = "binomial"
        self.family_params = {"n": n}
        self.dtype = jnp.float32

        # cache_key = f'samples-{seed}-{max_K}-{n}-{self.dtype}'
        # if cache_key in cache:
        #     self.samples = cache[cache_key]
        # else:
        #     key = jax.random.PRNGKey(seed)
        #     self.samples = jax.random.uniform(key, shape=(max_K, n), dtype=self.dtype)
        #     cache.update({cache_key: self.samples})
        #
        self.samples = cache(unifs)(seed, shape=(max_K, n), dtype=self.dtype)

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
