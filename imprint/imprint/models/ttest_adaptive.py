import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _sim(scaled_Z, B_unscaled, N, mu0, theta, null_truth):
    sigma_sq = -0.5 / theta[:, 1]
    sigma = jnp.sqrt(sigma_sq)
    mu = theta[:, 0] * sigma_sq
    A_div_N = sigma[:, None, None] * (scaled_Z / N[None])[None] + mu[:, None, None]
    B_div = sigma[:, None, None] * jnp.sqrt(B_unscaled / (N[None] - 1))[None]
    Ts = (A_div_N - (mu[:, None, None] - mu0)) / B_div
    return jnp.where(null_truth[:, None, 0], -jnp.max(Ts, axis=-1), jnp.inf)


class TTest1DAda:
    def __init__(self, seed, max_K, n_init, n_samples_per_interim, n_interims, mu0):
        n_samples_per_interim = np.array(n_samples_per_interim)
        if len(n_samples_per_interim.shape) == 0:
            n_samples_per_interim = np.full((n_interims,), n_samples_per_interim)
        self.n_samples_per_stage = np.concatenate([[n_init], n_samples_per_interim])
        self.N = np.cumsum(self.n_samples_per_stage)
        self.n_samples = self.N[-1]
        self.n_stages = len(self.N)

        self.family = "normal2"
        self.family_params = {"n": self.n_samples}
        self.dtype = jnp.float32
        self.mu0 = mu0
        self.n_interims = n_interims

        key = jax.random.PRNGKey(seed)
        normals = jax.random.normal(key, shape=(max_K, self.n_stages))
        self.scaled_Z = jnp.cumsum(jnp.sqrt(self.n_samples_per_stage) * normals, axis=1)

        _, key = jax.random.split(key)
        df = self.n_samples_per_stage - 1
        chisqs = 2 * jax.random.gamma(key, df / 2, shape=(max_K, self.n_stages))

        correction = (self.n_samples_per_stage[1:] * self.N[:-1] / self.N[1:])[None] * (
            normals[:, 1:] / jnp.sqrt(self.n_samples_per_stage[None, 1:])
            - self.scaled_Z[:, :-1] / self.N[None, :-1]
        ) ** 2
        correction = jnp.concatenate(
            [jnp.zeros(correction.shape[0])[:, None], correction], axis=1
        )
        self.B_unscaled = jnp.cumsum(chisqs + correction, axis=1)

    def sim_batch(self, begin_sim, end_sim, theta, null_truth):
        return _sim(
            self.scaled_Z[begin_sim:end_sim],
            self.B_unscaled[begin_sim:end_sim],
            self.N,
            self.mu0,
            theta,
            null_truth,
        )
