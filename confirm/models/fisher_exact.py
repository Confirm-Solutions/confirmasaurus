import jax
import jax.numpy as jnp


def hypergeom_logpmf(k, M, n, N):
    # Copied from scipy.stats.hypergeom
    tot, good = M, n
    bad = tot - good
    betaln = jax.scipy.special.betaln
    result = (
        betaln(good + 1, 1)
        + betaln(bad + 1, 1)
        + betaln(tot - N + 1, N + 1)
        - betaln(k + 1, good - k + 1)
        - betaln(N - k + 1, bad - N + k + 1)
        - betaln(tot + 1, 1)
    )
    return result


def hypergeom_logcdf(k, M, n, N):
    return jax.lax.fori_loop(
        1,
        k + 1,
        lambda i, acc: jax.scipy.special.logsumexp(
            jnp.array([acc, hypergeom_logpmf(i, M, n, N)])
        ),
        hypergeom_logpmf(0, M, n, N),
    )


def hypergeom_cdf(k, M, n, N):
    return jnp.exp(hypergeom_logcdf(k, M, n, N))


class FisherExact:
    def __init__(self, seed, max_K, *, n_arm_samples):
        self.family = "binomial"
        self.family_params = {"n": n_arm_samples}
        self.dtype = jnp.float32
        self.n_arm_samples = n_arm_samples

        # sample normals and then compute the CDF to transform into the
        # interval [0, 1]
        key = jax.random.PRNGKey(seed)
        self.samples = jax.random.uniform(
            key, shape=(max_K, self.n_arm_samples, 2), dtype=self.dtype
        )

    @staticmethod
    @jax.jit
    def _sim(samples, theta, null_truth):
        n = samples.shape[1]
        p = jax.scipy.special.expit(theta)
        successes = jnp.sum(samples[None, :] < p[:, None, None], axis=2)
        cdfvv = jax.vmap(
            jax.vmap(hypergeom_cdf, in_axes=(0, None, 0, None)),
            in_axes=(0, None, 0, None),
        )
        cdf = cdfvv(successes[..., 1], 2 * n, successes.sum(axis=-1), n)
        return 1 - cdf

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return self._sim(self.samples[begin_sim:end_sim], theta, null_truth)
