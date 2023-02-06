import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from imprint import batching
from imprint import driver
from imprint import grid


class BootstrapCalibrate:
    """
    Driver classes are the layer of imprint that is directly responsible for
    asking the Model for simulations.

    This driver has two entrypoints:
    1. `bootstrap_calibrate(...)`: calibrates (calculating lambda*) for every tile in a
       grid. The calibration is performed for many bootstrap resamplings of the
       simulated test statistics. This bootstrap gives a picture of the
       distribution of lambda*.
    2. `many_rej(...)`: calculates the number of rejections for many different
       values values of lambda*.

    For the basic `validate` and `calibrate` drivers, see `imprint.driver`.
    """

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
                jax.vmap(driver.calc_calibration_threshold, in_axes=(None, 0, None)),
                in_axes=(0, None, 0),
            )
        )

        self.tile_batch_size = tile_batch_size

        self.Ks = Ks
        self.nB = nB
        np.random.seed(bootstrap_seed)
        self.bootstrap_idxs = {
            K: np.concatenate(
                (
                    np.arange(K)[None, :],
                    np.sort(
                        np.random.choice(K, size=(self.nB + self.nB, K), replace=True),
                        axis=-1,
                    ),
                )
            ).astype(np.int32)
            for K in Ks
        }

        # Sampling using JAX is substantially slower on CPU than numpy.
        # I'm unsure what the performance ratio is like on GPU.
        # But, fast sampling for small nB and K is important for rapid
        # development, debugging and testing. I'm leaving this here so that
        # we can easily switch back to JAX sampling if we need to.
        #
        # bootstrap_key = jax.random.PRNGKey(bootstrap_seed)
        # self.bootstrap_idxs = {
        #     K: jnp.concatenate(
        #         (
        #             jnp.arange(K)[None, :],
        #             jnp.sort(
        #                 jax.random.choice(
        #                     bootstrap_key,
        #                     K,
        #                     shape=(self.nB + self.nB, K),
        #                     replace=True,
        #                 ),
        #                 axis=-1,
        #             ),
        #         )
        #     ).astype(jnp.int32)
        #     for K in Ks
        # }

    def bootstrap_calibrate(self, df, alpha, tile_batch_size=None):
        tile_batch_size = tile_batch_size or self.tile_batch_size

        def _batched(K, theta, vertices, null_truth):
            # NOTE: sort of a todo. having the simulations loop as the outer batch
            # is faster than the other way around. But, we need all simulations in
            # order to calibrate. So, the likely fastest solution is to have multiple
            # layers of batching. This is not currently implemented.
            stats = self.model.sim_batch(0, K, theta, null_truth)
            sorted_stats = jnp.sort(stats, axis=-1)
            alpha0 = self.backward_boundv(
                np.full(theta.shape[0], alpha), theta, vertices
            )
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
            lams_df.insert(0, "idx", driver._calibration_index(K, alpha0))
            lams_df.insert(0, "alpha0", alpha0)
            lams_df.insert(0, "K", K)

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

        out = driver._groupby_apply_K(df, f)
        out.insert(0, "K", df["K"])
        return out


def bootstrap_calibrate(
    modeltype,
    *,
    g: grid.Grid,
    alpha: float = 0.025,
    model_seed: int = 0,
    bootstrap_seed: int = 0,
    nB: int = 50,
    K: int = None,
    tile_batch_size: int = 64,
    model_kwargs: dict = None,
):
    """
    Calibrate the critical threshold for a given level of Type I Error control.

    Additionally, repeat this calibration for a two sets of bootstrap
    resamplings of the simulations:
    - the B (aka "bias") bootstrap resamples the simulations in order to
      estimate the bias in the calibrated lambda** (min of lambda*).
    - the twb (aka "tile-wise bootstrap") resamples the simulations in order
      to estimate the variance in lambda*

    The motivation for having these two bootstraps comes from the Adagrid
    algorithm where the bias bootstrap is used to estimate bias for the
    convergence criterion whereas the twb bootstrap is used to decide whether
    to add more simulations to a tile.

    Args:
        modeltype: The model class.
        g: The grid.
        alpha: The Type I Error control level. Defaults to 0.025.
        model_seed: The random seed. Defaults to 0.
        bootstrap_seed: The seed used for drawing the bootstrap resamples.
            Defaults to 0.
        nB: The number of resamples for both the bias and the tilewise
            bootstraps. That is to say, the total number of resamplings will be
            2*nB. Defaults to 50.
        K: The number of simulations. If this is unspecified, it is assumed
           that the grid has a "K" column containing per-tile simulation counts.
           Defaults to None.
        tile_batch_size: The number of tiles to simulate in a single batch.
        model_kwargs: Keyword arguments passed to the model constructor.
           Defaults to None.

    Returns:
        A dataframe with one row for each tile containing the columns:
        - lams: The calibrated lambda* from the original sample.
        - B_lams{i}: nB columns containing the calibrated lambda* from the
            bias bootstrap resamples.
        - twb_lams{i}: nB columns containing the calibrated lambda* from the
            tilewise bootstrap resamples.
        - twb_min_lams: The minimum of the twb_lams{i} columns.
        - twb_mean_lams: The mean of the twb_lams{i} columns.
        - twb_max_lams: The maximum of the twb_lams{i} columns.
        - alpha0: The alpha0 value used to calibrate the lambda*.
    """
    model, g = driver._setup(modeltype, g, model_seed, K, model_kwargs)
    Ks = np.sort(g.df["K"].unique())
    cal_df = BootstrapCalibrate(
        model, bootstrap_seed, nB, Ks, tile_batch_size
    ).bootstrap_calibrate(g.df, alpha)
    return cal_df
