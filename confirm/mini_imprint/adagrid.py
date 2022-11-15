import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import driver
from . import newlib


def tune(sorted_stats, sorted_order, alpha):
    K = sorted_stats.shape[0]
    cv_idx = jnp.maximum(jnp.floor((K + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0)
    # indexing a sorted array with sorted indices results in a sorted array!!
    return sorted_stats[sorted_order[cv_idx]]


class AdagridDriver:
    def __init__(self, model, init_K, n_K_double, nB, bootstrap_seed):
        self.model = model
        self.forward_boundv, self.backward_boundv = driver.get_bound(model)

        self.tunev = jax.jit(
            jax.vmap(jax.vmap(tune, in_axes=(None, 0, None)), in_axes=(0, None, 0))
        )

        bootstrap_key = jax.random.PRNGKey(bootstrap_seed)
        sim_sizes = init_K * 2 ** np.arange(n_K_double + 1)
        self.nB = nB
        self.bootstrap_idxs = {
            K: jnp.concatenate(
                (
                    jnp.arange(K)[None, :],
                    jnp.sort(
                        jax.random.choice(
                            bootstrap_key, K, shape=(nB + nB, K), replace=True
                        ),
                        axis=-1,
                    ),
                )
            ).astype(jnp.int32)
            for K in sim_sizes
        }

    def _bootstrap_tune(self, K_df, g, alpha=0.025):
        K = K_df["K"].iloc[0]
        assert all(K_df["K"] == K)
        K_g = newlib.Grid(g.d, K_df, g.null_hypos)

        theta, vertices = K_g.get_theta_and_vertices()
        alpha0 = self.backward_boundv(0.025, theta, vertices)
        # TODO: batching
        stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
        sorted_stats = jnp.sort(stats, axis=-1)
        bootstrap_lams = self.tunev(sorted_stats, self.bootstrap_idxs[K], alpha0)
        cols = ["lams"]
        for i in range(self.nB):
            cols.append(f"B_lams{i}")
        for i in range(self.nB):
            cols.append(f"twb_lams{i}")
        lams_df = pd.DataFrame(bootstrap_lams, index=K_df.index, columns=cols)
        lams_df.insert(0, "alpha0", alpha0)
        lams_df["twb_min_lams"] = bootstrap_lams[:, 1 + self.nB :].min(axis=1)
        lams_df["twb_mean_lams"] = bootstrap_lams[:, 1 + self.nB :].mean(axis=1)
        lams_df["twb_max_lams"] = bootstrap_lams[:, 1 + self.nB :].max(axis=1)
        return lams_df

    def bootstrap_tune(self, g, alpha=0.025):
        return g.df.groupby("K", group_keys=False).apply(
            lambda df: self._bootstrap_tune(df, g, alpha)
        )

    # def calc_bias(self, g, df_tune):
    #     lamss_idx = df_tune["lams"].argmin()
    #     lamss = df_tune["lams"].iloc[lamss_idx]
    #     B_lamss = df_tune[[f"B_lams{i}" for i in range(self.nB)]].values.min(axis=0)
    #     lams_bias = lamss - B_lamss.mean()

    #     K = g.df["K"].iloc[lamss_idx]
    #     theta = g.get_theta(slice(lamss_idx, lamss_idx + 1))
    #     stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())


# class Criterion