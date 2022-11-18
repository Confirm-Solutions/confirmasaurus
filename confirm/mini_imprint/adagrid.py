import copy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import db
from . import driver
from . import grid


class AdagridDriver:
    def __init__(self, model, *, init_K, n_K_double, nB, bootstrap_seed):
        self.model = model
        self.forward_boundv, self.backward_boundv = driver.get_bound(model)

        self.tunevv = jax.jit(
            jax.vmap(
                jax.vmap(driver.calc_tuning_threshold, in_axes=(None, 0, None)),
                in_axes=(0, None, 0),
            )
        )

        bootstrap_key = jax.random.PRNGKey(bootstrap_seed)
        self.Ks = init_K * 2 ** np.arange(n_K_double + 1)
        self.max_K = self.Ks[-1]
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
            for K in self.Ks
        }

    def _bootstrap_tune(self, K_df, alpha):
        K = K_df["K"].iloc[0]
        assert all(K_df["K"] == K)
        K_g = grid.Grid(K_df)

        theta, vertices = K_g.get_theta_and_vertices()
        alpha0 = self.backward_boundv(alpha, theta, vertices)
        # TODO: batching
        stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
        sorted_stats = jnp.sort(stats, axis=-1)
        bootstrap_lams = self.tunevv(sorted_stats, self.bootstrap_idxs[K], alpha0)
        cols = ["lams"]
        for i in range(self.nB):
            cols.append(f"B_lams{i}")
        for i in range(self.nB):
            cols.append(f"twb_lams{i}")
        lams_df = pd.DataFrame(bootstrap_lams, index=K_df.index, columns=cols)
        lams_df["twb_min_lams"] = bootstrap_lams[:, 1 + self.nB :].min(axis=1)
        lams_df["twb_mean_lams"] = bootstrap_lams[:, 1 + self.nB :].mean(axis=1)
        lams_df["twb_max_lams"] = bootstrap_lams[:, 1 + self.nB :].max(axis=1)

        lams_df.insert(0, "alpha0", alpha0)
        return lams_df

    def bootstrap_tune(self, df, alpha):
        return df.groupby("K", group_keys=False).apply(
            lambda K_df: self._bootstrap_tune(K_df, alpha)
        )

    def _many_rej(self, K_df, lams_arr):
        K = K_df["K"].iloc[0]
        K_g = grid.Grid(K_df)
        theta = K_g.get_theta()
        stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
        TI_sum = jnp.sum(stats[..., None] < lams_arr[None, None, :], axis=1)
        return TI_sum

    def many_rej(self, df, lams_arr):
        return df.groupby("K", group_keys=False).apply(
            lambda K_df: self._many_rej(K_df, lams_arr)
        )


class Adagrid:
    def __init__(
        self,
        ada_driver,
        g,
        *,
        db_type,
        grid_target,
        bias_target,
        std_target,
        iter_size,
        alpha,
        tuning_min_idx,
    ):
        self.ada_driver = ada_driver
        self.alpha = alpha
        self.tuning_min_idx = tuning_min_idx
        self.bias_target = bias_target
        self.grid_target = grid_target
        self.std_target = std_target
        self.iter_size = iter_size

        self.null_hypos = g.null_hypos
        g_tuned = self.process_tiles(g, 0)
        self.tiledb = db_type.create(g_tuned.df)

    def process_tiles(self, g, i):
        lams_df = self.ada_driver.bootstrap_tune(g.df, self.alpha)
        # we use insert here to order columns nicely for reading raw data
        lams_df.insert(1, "grid_cost", self.alpha - lams_df["alpha0"])
        lams_df.insert(
            2,
            "impossible",
            lams_df["alpha0"] < (self.tuning_min_idx + 1) / (g.df["K"] + 1),
        )
        lams_df.insert(
            3,
            "orderer",
            np.minimum(
                lams_df["twb_min_lams"],
                np.where(lams_df["impossible"], -np.inf, np.inf),
            ),
        )
        g_tuned = g.add_cols(lams_df)
        g_tuned.df["birthday"] = i
        return g_tuned

    def step(self, i):
        worst_tile = self.tiledb.worst_tile("lams")
        lamss = worst_tile["lams"].iloc[0]
        B_lamss = self.tiledb.bootstrap_lamss()

        worst_tile_TI_sum = self.ada_driver.many_rej(
            worst_tile, np.array([lamss] + list(B_lamss))
        ).iloc[0][0]

        worst_tile_TI_est = worst_tile_TI_sum / worst_tile["K"].iloc[0]
        bias_tie = worst_tile_TI_est[0] - worst_tile_TI_est[1:].mean()
        std_tie = worst_tile_TI_est.std()
        spread_tie = worst_tile_TI_est.max() - worst_tile_TI_est.min()
        grid_cost = worst_tile["grid_cost"].iloc[0]

        self.tiledb.unlock_all()
        work = self.tiledb.next(self.iter_size, "orderer")

        twb_worst_tile = self.tiledb.worst_tile("twb_mean_lams")
        twb_worst_tile_g = grid.Grid(twb_worst_tile)
        for d in range(twb_worst_tile_g.d):
            twb_worst_tile[f"radii{d}"] = 1e-6
        twb_worst_tile_lams = self.ada_driver.bootstrap_tune(twb_worst_tile, self.alpha)
        twb_worst_tile_mean_lams = twb_worst_tile_lams["twb_mean_lams"].iloc[0]
        deepen_likely_to_work = work["twb_mean_lams"] > twb_worst_tile_mean_lams
        work["refine"] = work["grid_cost"] > self.grid_target
        work["deepen"] = (deepen_likely_to_work | (~work["refine"])) & (
            work["K"] < self.ada_driver.max_K
        )
        work["refine"] &= ~work["deepen"]
        work["active"] = ~(work["refine"] | work["deepen"])

        n_refine = work["refine"].sum()
        n_deepen = work["deepen"].sum()
        if n_refine > 0 or n_deepen > 0:
            g_deepen_in = grid.Grid(work.loc[work["deepen"]])
            g_deepen = grid.init_grid(
                g_deepen_in.get_theta(),
                g_deepen_in.get_radii(),
                g_deepen_in.df["K"] * 2,
                g_deepen_in.df["id"],
            )
            g_refine = grid.Grid(work.loc[work["refine"]]).refine()
            g_new = g_refine.concat(g_deepen).add_null_hypos(self.null_hypos).prune()
            g_tuned_new = self.process_tiles(g_new, i)
            self.tiledb.write(g_tuned_new.df)
        self.tiledb.finish(work)

        report = dict(
            i=i,
            bias_tie=bias_tie,
            std_tie=std_tie,
            spread_tie=spread_tie,
            grid_cost=grid_cost,
            n_refine=n_refine,
            n_deepen=n_deepen,
            n_finished=work["active"].sum(),
            n_impossible=work["impossible"].sum(),
            lamss=lamss,
            B_lamss_min=min(B_lamss),
            B_lamss_max=max(B_lamss),
        )
        report["tie_{k}(lamss)"] = worst_tile_TI_est[0]

        done = (
            (bias_tie < self.bias_target)
            and (grid_cost < self.grid_target)
            and (std_tie < self.std_target)
        )
        return done, report


def print_report(_iter, report, _ada):
    from rich import print as rprint

    ready = report.copy()
    for k in ready:
        if isinstance(ready[k], float) or isinstance(ready[k], jnp.DeviceArray):
            ready[k] = f"{ready[k]:.6f}"
    rprint(ready)


def ada_tune(
    modeltype,
    g,
    *,
    model_seed=0,
    bootstrap_seed=0,
    init_K=2**13,
    n_K_double=4,
    nB=50,
    grid_target=0.001,
    bias_target=0.001,
    std_target=0.002,
    iter_size=2**10,
    alpha=0.025,
    tuning_min_idx=40,
    max_iter=100,
    db_type=db.DuckDBTiles,
    callback=print_report,
    model_kwargs=None,
):
    g = copy.deepcopy(g)
    g.df["K"] = init_K

    if model_kwargs is None:
        model_kwargs = {}
    model = modeltype(seed=model_seed, max_K=init_K * 2**n_K_double, **model_kwargs)

    ada_driver = AdagridDriver(
        model,
        init_K=init_K,
        n_K_double=n_K_double,
        nB=nB,
        bootstrap_seed=bootstrap_seed,
    )

    ada = Adagrid(
        ada_driver,
        g,
        db_type=db_type,
        grid_target=grid_target,
        bias_target=bias_target,
        std_target=std_target,
        iter_size=iter_size,
        alpha=alpha,
        tuning_min_idx=tuning_min_idx,
    )

    reports = []
    for ada_iter in range(1, max_iter):
        done, report = ada.step(ada_iter)
        if callback is not None:
            callback(ada_iter, report, ada)
        reports.append(report)
        if done:
            break
    return ada, reports
