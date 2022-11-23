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


class Binom1D:
    def __init__(self, seed, max_K, *, n):
        self.family = "binomial"
        self.family_params = {"n": n}
        self.dtype = jnp.float32

        # sample normals and then compute the CDF to transform into the
        # interval [0, 1]
        key = jax.random.PRNGKey(seed)
        self.samples = jax.random.uniform(key, shape=(max_K, n), dtype=self.dtype)
        print(self.samples.shape)

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
