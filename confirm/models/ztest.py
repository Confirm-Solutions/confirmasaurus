import jax
import jax.numpy as jnp
import pandas as pd


@jax.jit
def _sim(samples, theta, null_truth):
    return jnp.where(
        null_truth[:, None, 0],
        # negate so that we can do a less than comparison
        -(theta[:, None, 0] + samples[None, :]),
        jnp.inf,
    )


class ZTest1D:
    def __init__(self, seed, max_K, cache=None):
        self.family = "normal"
        self.dtype = jnp.float32

        # sample normals and then compute the CDF to transform into the
        # interval [0, 1]
        key = jax.random.PRNGKey(seed)

        key = f"ZTest1D-{seed}-{max_K}"
        if cache is not None and cache.contains(key):
            self.samples = cache.load(key)["samples"].values
        else:
            self.samples = jax.random.normal(key, shape=(max_K,), dtype=self.dtype)
            if cache is not None:
                cache.store(key, pd.DataFrame(dict(samples=self.samples)))

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
