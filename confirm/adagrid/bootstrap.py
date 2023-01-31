import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from imprint import batching
from imprint import driver
from imprint import grid


class BootstrapCalibrate:
    def __init__(
        self, model, bootstrap_seed, nB, Ks, tile_batch_size=64, worker_id=None
    ):
        self.model = model
        self.worker_id = worker_id
        self.forward_boundv, self.backward_boundv = driver.get_bound(
            model.family, model.family_params if hasattr(model, "family_params") else {}
        )
        self.calibratevv = jax.jit(
            jax.vmap(
                jax.vmap(driver.calc_tuning_threshold, in_axes=(None, 0, None)),
                in_axes=(0, None, 0),
            )
        )

        bootstrap_key = jax.random.PRNGKey(bootstrap_seed)
        self.Ks = Ks
        self.nB = nB
        self.bootstrap_idxs = {
            K: jnp.concatenate(
                (
                    jnp.arange(K)[None, :],
                    jnp.sort(
                        jax.random.choice(
                            bootstrap_key,
                            K,
                            shape=(self.nB + self.nB, K),
                            replace=True,
                        ),
                        axis=-1,
                    ),
                )
            ).astype(jnp.int32)
            for K in Ks
        }

        self.tile_batch_size = tile_batch_size

    def bootstrap_calibrate(self, df, alpha, tile_batch_size=None):
        tile_batch_size = tile_batch_size or self.tile_batch_size

        def _batched(K, theta, vertices, null_truth):
            # NOTE: sort of a todo. having the simulations loop as the outer batch
            # is faster than the other way around. But, we need all simulations in
            # order to calibrate. So, the likely fastest solution is to have multiple
            # layers of batching. This is not currently implemented.
            stats = self.model.sim_batch(0, K, theta, null_truth)
            sorted_stats = jnp.sort(stats, axis=-1)
            alpha0 = self.backward_boundv(alpha, theta, vertices)
            return (
                self.calibratevv(sorted_stats, self.bootstrap_idxs[K], alpha0),
                alpha0,
            )

        def f(K, K_df):
            K_g = grid.Grid(K_df, self.worker_id)

            theta, vertices = K_g.get_theta_and_vertices()
            bootstrap_lams, alpha0 = batching.batch(
                _batched,
                self.tile_batch_size,
                in_axes=(None, 0, 0, 0),
            )(K, theta, vertices, K_g.get_null_truth())

            cols = ["lams"]
            for i in range(self.nB):
                cols.append(f"B_lams{i}")
            for i in range(self.nB):
                cols.append(f"twb_lams{i}")
            lams_df = pd.DataFrame(bootstrap_lams, index=K_df.index, columns=cols)

            lams_df.insert(
                0, "twb_min_lams", bootstrap_lams[:, 1 + self.nB :].min(axis=1)
            )
            lams_df.insert(
                0, "twb_mean_lams", bootstrap_lams[:, 1 + self.nB :].mean(axis=1)
            )
            lams_df.insert(
                0, "twb_max_lams", bootstrap_lams[:, 1 + self.nB :].max(axis=1)
            )
            lams_df.insert(0, "alpha0", alpha0)

            return lams_df

        return driver._groupby_apply_K(df, f)

    def many_rej(self, df, lams_arr):
        def f(K, K_df):
            K_g = grid.Grid(K_df, self.worker_id)

            theta = K_g.get_theta()

            # NOTE: no batching implemented here. Currently, this function is only
            # called with a single tile so it's not necessary.
            stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
            tie_sum = jnp.sum(stats[..., None] < lams_arr[None, None, :], axis=1)
            return pd.DataFrame(
                tie_sum,
                index=K_df.index,
                columns=[str(i) for i in range(lams_arr.shape[0])],
            )

        return driver._groupby_apply_K(df, f)


def bootstrap_calibrate(
    modeltype,
    g: grid.Grid,
    *,
    model_seed: int = 0,
    bootstrap_seed: int = 0,
    nB: int = 50,
    alpha: float = 0.025,
    K: int = None,
    tile_batch_size: int = 64,
    model_kwargs: dict = None,
):
    model, g = driver._setup(modeltype, g, model_seed, K, model_kwargs)
    Ks = np.sort(g.df["K"].unique())
    cal_df = BootstrapCalibrate(
        model, bootstrap_seed, nB, Ks, tile_batch_size
    ).bootstrap_calibrate(g.df, alpha)
    return cal_df
