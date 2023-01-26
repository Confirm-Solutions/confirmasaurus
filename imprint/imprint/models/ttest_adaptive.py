# import jax
import jax.numpy as jnp

# from . import ttest


class TTest1DAda:
    def __init__(self, seed, max_K, n_samples_per_interim, n_interims, mu0):
        self.n_samples = n_samples_per_interim * n_interims
        self.family = "normal2"
        self.family_params = {"n": self.n_samples}
        self.dtype = jnp.float32
        self.mu0 = mu0

    def sim_batch(self, begin_sim, end_sim, theta, null_truth):
        pass
