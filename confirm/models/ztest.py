import jax
import jax.numpy as jnp

from confirm.mini_imprint import batch


@jax.jit
def _sim(samples, theta, null_truth):
    return jnp.where(
        null_truth[:, None, 0],
        # negate so that we can do a less than comparison
        -(theta[:, None, 0] + samples[None, :]),
        jnp.inf,
    )


class ZTest1D:
    def __init__(self, seed, max_K, *, sim_batch_size=2048):
        self.family = "normal"
        self.sim_batch_size = sim_batch_size
        self.dtype = jnp.float32

        # sample normals and then compute the CDF to transform into the
        # interval [0, 1]
        key = jax.random.PRNGKey(seed)
        self.samples = jax.random.normal(key, shape=(max_K,), dtype=self.dtype)
        self._sim_batch = batch(
            _sim, self.sim_batch_size, in_axes=(0, None, None), out_axes=(1,)
        )

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return self._sim_batch(self.samples[begin_sim:end_sim], theta, null_truth)


# def main():
#     import confirm.mini_imprint as ip
#     import scipy.stats
#     import numpy as np

#     g = ip.cartesian_grid([-1], [1], n=[10], null_hypos=[ip.hypo("x < 0")])
#     # lam = -1.96 because we negated the statistics so we can do a less thanj
#     # comparison.
#     lam = -1.96
#     rej_df = ip.validate(ZTest1D, g, lam, K=8192)
#     true_err = 1 - scipy.stats.norm.cdf(-g.get_theta()[:, 0] - lam)

#     tie_est = rej_df["TI_sum"] / rej_df["K"]
#     tie_std = scipy.stats.binom.std(n=K, p=true_err) / K
#     n_stds = (tie_est - true_err) / tie_std

#     tune_df = ip.tune(ZTest1D, g)


# if __name__ == "__main__":
#     main()
