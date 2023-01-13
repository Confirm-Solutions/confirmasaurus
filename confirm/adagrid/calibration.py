"""
Broadly speaking, the adaptive calibration algorithm refines and deepens tiles until
the type I error, max(f_hat(lambda**)), is close to the type I error control
level, alpha.

The core tool to decide whether a tile has a sufficient number of simulations
is to calculate lambda* from several bootstrap resamples of the test
statistics. Deciding whether a tile needs to be refined is easier because we
can directly calculate the type I error slack from continuous simulation
extension: alpha - alpha0

The remaining tricky part then is deciding which tiles to refine/deepen
*first*! We primarily do this based on a second bootstrap resampling of
lambda*. This second bootstrap is called the "twb" bootstrap. The tiles with
the smallest min_{twb}(lambda*) are processed first.
TODO: we would like to inflate these minima.
TODO: discuss the issues of many near-critical tiles. minimum of gaussian
draws as a useful model.

See the comments in Adagrid.step for a detailed understanding of the algorithm.

The great glossary of adagrid:
- K: the number of simulations to use for this tile.
- tie: (T)ype (I) (E)rror
- alpha: the type I error control level. Typically 0.025.
- deepen: add more simulations. We normally double the number of simulations.
- refine: replace a tile with 2^d children that have half the radius.
- twb: "tilewise bootstrap"
- f_hat(lambda**): the type I error with lambda=lambda**
- B: "bias bootstrap". The set of bootstrap resamples used for computing the
  bias in f_hat(lambda**)
- lams aka "lambda*" "lambda star": The calibration threshold for a given tile.
- twb_lams, B_lams: the lambda star from each bootstrap for this tile.
- `twb_[min, mean, max]_lams`: The minimum lambda* for this tile across the twb
  bootstraps
- lamss aka "lambda**": The minimum calibration threshold over the whole grid.
- alpha0: backward_cse(this_tile, alpha)
- grid_cost: alpha - alpha0
- impossible tiles are those for which K and alpha0 are not large enough to
  satisfy the calibration_min_idx constraint.
- orderer: the value by which we decide the next tiles to "process"
- processing tiles: deciding to refine or deepen and then simulating for those
  new tiles that resulted from refining or deepening.
- worst tile: the tile for which lams is smallest. `lams[worst_tile] == lamss`
"""
import copy
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pprint import pprint

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .db import DuckDBTiles
from imprint import batching
from imprint import driver
from imprint import grid
from imprint.timer import simple_timer


class AdaCalibrationDriver:
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

    For the basic `validate` and `calibrate` drivers, see `driver.py`.
    """

    def __init__(self, db, model, null_hypos, config):
        self.db = db
        self.model = model
        self.null_hypos = null_hypos
        self.c = config
        self.forward_boundv, self.backward_boundv = driver.get_bound(
            model.family, model.family_params if hasattr(model, "family_params") else {}
        )

        self.calibratevv = jax.jit(
            jax.vmap(
                jax.vmap(driver.calc_tuning_threshold, in_axes=(None, 0, None)),
                in_axes=(0, None, 0),
            )
        )

        bootstrap_key = jax.random.PRNGKey(self.c.bootstrap_seed)
        self.init_K = self.c.init_K
        self.Ks = self.c.init_K * 2 ** np.arange(self.c.n_K_double + 1)
        self.max_K = self.Ks[-1]
        self.nB = self.c.nB
        self.bootstrap_idxs = {
            K: jnp.concatenate(
                (
                    jnp.arange(K)[None, :],
                    jnp.sort(
                        jax.random.choice(
                            bootstrap_key,
                            K,
                            shape=(self.c.nB + self.c.nB, K),
                            replace=True,
                        ),
                        axis=-1,
                    ),
                )
            ).astype(jnp.int32)
            for K in self.Ks
        }

        self.tile_batch_size = self.c.tile_batch_size

        self.alpha = self.c.alpha
        self.calibration_min_idx = self.c.calibration_min_idx
        self.bias_target = self.c.bias_target
        self.grid_target = self.c.grid_target
        self.std_target = self.c.std_target
        self.iter_size = self.c.iter_size

    def bootstrap_calibrate(self, df, alpha, tile_batch_size=None):
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

        def f(K_df):
            _tbs = self.tile_batch_size if tile_batch_size is None else tile_batch_size

            K = K_df["K"].iloc[0]
            assert all(K_df["K"] == K)
            K_g = grid.Grid(K_df)

            theta, vertices = K_g.get_theta_and_vertices()
            bootstrap_lams, alpha0 = batching.batch(
                _batched,
                _tbs,
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

        return df.groupby("K", group_keys=False).apply(f)

    def many_rej(self, df, lams_arr):
        def f(K_df):
            K = K_df["K"].iloc[0]
            K_g = grid.Grid(K_df)
            theta = K_g.get_theta()
            stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
            tie_sum = jnp.sum(stats[..., None] < lams_arr[None, None, :], axis=1)
            return tie_sum

        # NOTE: no batching implemented here. Currently, this function is only
        # called with a single tile so it's not necessary.
        return df.groupby("K", group_keys=False).apply(f)

    def _process_tiles(self, g, i):
        # This method actually runs the calibration and bootstrapping.
        # It is called once per iteration.
        # Several auxiliary fields are calculated because they are needed for
        # selecting the next iteration's tiles: impossible and orderer

        lams_df = self.bootstrap_calibrate(g.df, self.alpha)

        g.df["worker_id"] = self.c.worker_id
        g.df["birthiter"] = i
        g.df["birthtime"] = simple_timer()
        g.df["eligible"] = True

        # we use insert here to order columns nicely for reading raw data
        lams_df.insert(1, "grid_cost", self.alpha - lams_df["alpha0"])
        lams_df.insert(
            2,
            "impossible",
            lams_df["alpha0"] < (self.calibration_min_idx + 1) / (g.df["K"] + 1),
        )

        lams_df.insert(
            3,
            "orderer",
            # Where calibration is impossible due to either small K or small alpha0,
            # the orderer is set to -inf so that such tiles are guaranteed to
            # be processed.
            np.minimum(
                lams_df["twb_min_lams"],
                np.where(lams_df["impossible"], -np.inf, np.inf),
            ),
        )
        g_calibrated = g.add_cols(lams_df)
        return g_calibrated

    def convergence_criterion(self, any_impossible, report):
        ########################################
        # Step 2: Convergence criterion! In terms of:
        # - bias
        # - standard deviation
        # - grid cost (i.e. alpha - alpha0)
        #
        # The bias and standard deviation are calculated using the bootstrap.
        ########################################
        if any_impossible:
            return False

        worst_tile = self.db.worst_tile("lams")
        lamss = worst_tile["lams"].iloc[0]

        # We determine the bias by comparing the Type I error at the worst
        # tile for each lambda**_B:
        B_lamss = self.db.bootstrap_lamss()
        worst_tile_tie_sum = self.many_rej(
            worst_tile, np.array([lamss] + list(B_lamss))
        ).iloc[0][0]
        worst_tile_tie_est = worst_tile_tie_sum / worst_tile["K"].iloc[0]

        # Given these TIE values, we can compute bias, standard deviation and
        # spread.
        bias_tie = worst_tile_tie_est[0] - worst_tile_tie_est[1:].mean()
        std_tie = worst_tile_tie_est.std()
        spread_tie = worst_tile_tie_est.max() - worst_tile_tie_est.min()
        grid_cost = worst_tile["grid_cost"].iloc[0]

        report.update(
            dict(
                bias_tie=bias_tie,
                std_tie=std_tie,
                spread_tie=spread_tie,
                grid_cost=grid_cost,
                lamss=lamss,
            )
        )
        report["min(B_lamss)"] = min(B_lamss)
        report["max(B_lamss)"] = max(B_lamss)
        report["tie_{k}(lamss)"] = worst_tile_tie_est[0]
        report["tie + slack"] = worst_tile_tie_est[0] + grid_cost + bias_tie

        # The convergence criterion itself.
        report["converged"] = (
            (bias_tie < self.bias_target)
            and (std_tie < self.std_target)
            and (grid_cost < self.grid_target)
        )
        return report["converged"]

    def step(self, i):
        """
        One iteration of the adagrid algorithm.

        Args:
            i: The iteration number.

        Returns:
            done: True if the algorithm has converged.
            report: A dictionary of information about the iteration.
        """

        start_get_work = time.time()
        report = dict(i=i)

        ########################################
        # Step 1: Get the next batch of tiles to process. We do this before
        # checking for convergence because part of the convergence criterion is
        # whether there are any impossible tiles.
        ########################################
        with self.db.lock:
            packet_id = self.db.get_active_packet_id()
            work = self.db.next(packet_id, self.iter_size, "orderer", self.c.worker_id)
            report["n_impossible"] = work["impossible"].sum()
            report["runtime_get_work"] = time.time() - start_get_work

            if work.shape[0] == 0:
                if self.db.n_tiles_left_in_packet(packet_id) == 0:
                    start_convergence = time.time()
                    done = self.convergence_criterion(work["impossible"].any(), report)
                    report["runtime_convergence_check"] = (
                        time.time() - start_convergence
                    )
                    if done:
                        report["desc"] = "converged"
                        # TODO: stop other workers? or just let them figure it out?
                        return True, report
                    start_new_packet = time.time()
                    self.db.new_packet(packet_id + 1, self.packet_size, "orderer")
                    report["runtime_new_packet"] = time.time() - start_new_packet
                    report["desc"] = "new_packet"
                    return False, report
                else:
                    # TODO: should this be configurable?
                    time.sleep(5)
                    report["desc"] = "waiting"
                    return False, report
        report["desc"] = "working"

        ########################################
        # Step 4: Is deepening likely to be enough?
        ########################################

        # When we are deciding to refine or deepen, it's helpful to know
        # whether a tile is ever going to "important". That is, will the
        # tile ever be the worst tile?
        #
        # This is useful because deepening is normally cheaper than refinement.
        # In cases where it seems like a tile would be unimportant if variance
        # were reduced, then we can probably save effort by deepening instead
        # of refining.
        #
        # To answer whether a tile might ever be the worst tile, we compare the
        # given tile's bootstrapped mean lambda* against the bootstrapped mean
        # lambda* of the tile with the lowest mean lambda*
        # - recomputed with zero grid_cost (that is: alpha = alpha0)
        #   by specifying a tiny radius
        # - recomputed with the maximum allowed K
        #
        # If the tile's mean lambda* is less the mean lambda* of this modified
        # tile, then the tile actually has a chance of being the worst tile. In
        # which case, we choose the more expensive option of refining the tile.
        start_refine_deepen = time.time()
        twb_worst_tile = self.db.worst_tile("twb_mean_lams")
        for col in twb_worst_tile.columns:
            if col.startswith("radii"):
                twb_worst_tile[col] = 1e-6
        twb_worst_tile["K"] = self.max_K
        twb_worst_tile_lams = self.bootstrap_calibrate(
            twb_worst_tile, self.alpha, tile_batch_size=1
        )
        twb_worst_tile_mean_lams = twb_worst_tile_lams["twb_mean_lams"].iloc[0]
        deepen_likely_to_work = work["twb_mean_lams"] > twb_worst_tile_mean_lams

        ########################################
        # Step 5: Decide whether to refine or deepen
        # THIS IS IT! This is where we decide whether to refine or deepen.
        ########################################
        at_max_K = work["K"] == self.max_K
        work["refine"] = (work["grid_cost"] > self.grid_target) & (
            (~deepen_likely_to_work) | at_max_K
        )
        work["deepen"] = (~work["refine"]) & (~at_max_K)
        work["active"] = ~(work["refine"] | work["deepen"])

        ########################################
        # Step 6: Deepen and refine tiles.
        ########################################
        n_refine = work["refine"].sum()
        n_deepen = work["deepen"].sum()
        nothing_to_do = n_refine == 0 and n_deepen == 0
        if not nothing_to_do:
            g_new = refine_deepen(work, self.null_hypos, self.max_K)

            report["runtime_refine_deepen"] = time.time() - start_refine_deepen
            start_processing = time.time()
            g_calibrated_new = self._process_tiles(g_new, i)
            report["runtime_processing"] = time.time() - start_processing
            start_cleanup = time.time()
            self.db.write(g_calibrated_new.df)
            report.update(
                dict(
                    n_processed=g_calibrated_new.n_tiles,
                    K_distribution=g_calibrated_new.df["K"].value_counts().to_dict(),
                )
            )

        ########################################
        # Step 8: Finish up!
        ########################################
        # We need to report back to the TileDB that we're done with this batch
        # of tiles and whether any of the tiles are still active.
        self.db.finish(work)
        if nothing_to_do:
            report["runtime_cleanup"] = time.time() - start_refine_deepen
        else:
            report["runtime_cleanup"] = time.time() - start_cleanup

        report.update(
            dict(
                n_refine=n_refine,
                n_deepen=n_deepen,
                n_complete=work["active"].sum(),
            )
        )

        return False, report


def refine_deepen(g, null_hypos, max_K):
    g_deepen_in = grid.Grid(g.loc[g["deepen"] & (g["K"] < max_K)])
    g_deepen = grid.init_grid(
        g_deepen_in.get_theta(),
        g_deepen_in.get_radii(),
        g_deepen_in.df["id"],
    )

    # We just multiply K by 2 to deepen.
    # TODO: it's possible to do better by multiplying by 4 or 8
    # sometimes when a tile clearly needs *way* more sims. how to
    # determine this?
    g_deepen.df["K"] = g_deepen_in.df["K"] * 2

    g_refine_in = grid.Grid(g.loc[g["refine"]])
    inherit_cols = ["K"]
    # TODO: it's possible to do better by refining by more than just a
    # factor of 2.
    g_refine = g_refine_in.refine(inherit_cols)

    ########################################
    # Step 7: Simulate the new tiles and write to the DB.
    ########################################
    return g_refine.concat(g_deepen).add_null_hypos(null_hypos, inherit_cols).prune()


def print_report(_iter, report, _ada):
    ready = report.copy()
    for k in ready:
        if isinstance(ready[k], float) or isinstance(ready[k], jnp.DeviceArray):
            ready[k] = f"{ready[k]:.6f}"
    pprint(ready)


def _run(cmd):
    try:
        return (
            subprocess.check_output(" ".join(cmd), stderr=subprocess.STDOUT, shell=True)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.returncode} {exc.output}"


def _get_git_revision_hash() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def _get_git_diff() -> str:
    return _run(["git", "diff", "HEAD"])


def _get_nvidia_smi() -> str:
    return _run(["nvidia-smi"])


def _get_pip_freeze() -> str:
    from pip._internal.operations import freeze

    pkgs = freeze.freeze()
    return "\n".join(list(pkgs))


def _get_conda_list() -> str:
    return _run(["conda", "list"])


calibration_defaults = dict(
    model_name="invalid",
    model_seed=0,
    model_kwargs=None,
    alpha=0.025,
    init_K=2**13,
    n_K_double=4,
    bootstrap_seed=0,
    nB=50,
    tile_batch_size=None,
    grid_target=0.001,
    bias_target=0.001,
    std_target=0.002,
    calibration_min_idx=40,
    packet_size=2**12,
    iter_size=2**9,
    n_iter=100,
    prod=True,
    worker_id=None,
    git_hash=None,
    git_diff=None,
    nvidia_smi=None,
    pip_freeze=None,
    conda_list=None,
    platform=None,
)


@dataclass
class CalibrationConfig:
    """
    CalibrationConfig is a dataclass that holds all the configuration for a
    calibration run. For each worker, the data here will be written to the
    database so that we can keep track of what was run and how.
    """

    modeltype: type
    model_seed: int
    model_kwargs: dict
    alpha: float
    init_K: int
    n_K_double: int
    bootstrap_seed: int
    nB: int
    tile_batch_size: int
    grid_target: float
    bias_target: float
    std_target: float
    calibration_min_idx: int
    packet_size: int
    iter_size: int
    n_iter: int
    prod: bool
    worker_id: int
    git_hash: str = None
    git_diff: str = None
    nvidia_smi: str = None
    pip_freeze: str = None
    conda_list: str = None
    platform: str = None
    defaults: dict = None

    def __post_init__(self):
        self.git_hash = _get_git_revision_hash()
        self.git_diff = _get_git_diff()
        self.platform = platform.platform()
        self.nvidia_smi = _get_nvidia_smi()
        if self.prod:
            self.pip_freeze = _get_pip_freeze()
            self.conda_list = _get_conda_list()
        else:
            self.pip_freeze = "skipped for non-prod run"
            self.conda_list = "skipped for non-prod run"

        self.tile_batch_size = self.tile_batch_size or (
            64 if jax.lib.xla_bridge.get_backend().platform == "gpu" else 4
        )
        self.model_name = self.modeltype.__name__  # noqa
        if self.model_kwargs is None:
            self.model_kwargs = {}
        # TODO: is json suitable for all models? are there models that are going to
        # want to have large non-jsonable objects as parameters?
        self.model_kwargs = json.dumps(self.model_kwargs)

        continuation = True
        if self.defaults is None:
            self.defaults = calibration_defaults
            continuation = False

        for k in self.defaults:
            if self.__dict__[k] is None:
                self.__dict__[k] = self.defaults[k]

        # If we're continuing a calibration, make sure that fixed parameters
        # are the same across all workers.
        if continuation:
            for k in [
                "model_seed",
                "model_kwargs",
                "alpha",
                "init_K",
                "n_K_double",
                "bootstrap_seed",
                "nB",
                "model_name",
            ]:
                if self.__dict__[k] != self.defaults[k]:
                    raise ValueError(
                        f"Fixed parameter {k} has different values across workers."
                    )

        config_dict = {k: self.__dict__[k] for k in self.defaults}
        self.config_df = pd.DataFrame([config_dict])


def _load_null_hypos(db):
    d = db.dimension()
    null_hypos_df = db.store.get("null_hypos")
    null_hypos = []
    for i in range(null_hypos_df.shape[0]):
        n = np.array([null_hypos_df[f"n{i}"].iloc[i] for i in range(d)])
        c = null_hypos_df["c"].iloc[i]
        null_hypos.append(grid.HyperPlane(n, c))
    return null_hypos


def _store_null_hypos(db, null_hypos):
    d = db.dimension()
    n_hypos = len(null_hypos)
    cols = {f"n{i}": [null_hypos[j].n[i] for j in range(n_hypos)] for i in range(d)}
    cols["c"] = [null_hypos[j].c for j in range(n_hypos)]
    null_hypos_df = pd.DataFrame(cols)
    db.store.set("null_hypos", null_hypos_df)


def ada_calibrate(
    modeltype,
    *,
    g=None,
    db=None,
    model_seed: int = None,
    model_kwargs=None,
    alpha: float = None,
    init_K: int = None,
    n_K_double: int = None,
    bootstrap_seed: int = None,
    nB: int = None,
    tile_batch_size: int = None,
    grid_target: float = None,
    bias_target: float = None,
    std_target: float = None,
    calibration_min_idx: int = None,
    packet_size: int = None,
    iter_size: int = None,
    n_iter: int = None,
    prod: bool = True,
    callback=print_report,
):
    """
    The main entrypoint for the adaptive calibration algorithm.

    Args:
        modeltype: The model class to use.
        g: The initial grid.
        db: The database backend to use. Defaults to `db.DuckDB.connect()`.
        model_seed: The random seed for the model. Defaults to 0.
        model_kwargs: Additional keyword arguments for constructing the Model
                  object. Defaults to None.
        alpha: The target type I error control level. Defaults to 0.025.
        init_K: Initial K for the first tiles. Defaults to 2**13.
        n_K_double: The number of doublings of K. The maximum K will be
                    `init_K * 2 ** (n_K_double + 1)`. Defaults to 4.
        bootstrap_seed: The random seed for bootstrapping. Defaults to 0.
        nB: The number of bootstrap samples. Defaults to 50.
        tile_batch_size: The number of tiles to simulate in a single batch.
            Defaults to 64 on GPU and 4 on CPU.
        grid_target: Part of the stopping criterion: the target slack from CSE.
                     Defaults to 0.001.
        bias_target: Part of the stopping criterion: the target bias as
                     calculated by the bootstrap. Defaults to 0.001.
        std_target: Part of the stopping criterion: the target standard
                    deviation of the type I error as calculated by the
                    bootstrap. Defaults to 0.002.
        packet_size: The number of tiles in a "packet" produced by a single
                     Adagrid tile selection step. This is different from
                     iter_size because we select tiles once and then run many
                     "iterations" in parallel to process those tiles.
        iter_size: The number of tiles to process per iteration. Defaults to 2**10.
        calibration_min_idx: The minimum calibration selection index. We enforce that:
                        `alpha0 >= (calibration_min_idx + 1) / (K + 1)`
                        A larger value will reduce the variance of lambda* but
                        will require more computational effort because K and/or
                        alpha0 will need to be larger. Defaults to 40.
        n_iter: The number of adagrid iterations to run.
                Defaults to 100.
        prod: Is this a production run? If so, we will collection extra system
              configuration info.
        callback: A function accepting three arguments (ada_iter, report, ada)
                  that can perform some reporting or printing at each iteration.
                  Defaults to print_report.

    Returns:
        ada_iter: The final iteration number.
        reports: A list of the report dicts from each iteration.
        ada: The Adagrid object after the final iteration.
    """

    ########################################
    # Setup the database, load
    ########################################
    if g is None and db is None:
        raise ValueError("Must provide either an initial grid or a database!")

    if db is None:
        db = DuckDBTiles.connect()
    worker_id = db.new_worker()

    ########################################
    # Store config
    ########################################

    defaults = None
    if g is None:
        defaults = db.store.get("config").iloc[0].to_dict()

    c = CalibrationConfig(
        modeltype,
        model_seed,
        model_kwargs,
        alpha,
        init_K,
        n_K_double,
        bootstrap_seed,
        nB,
        tile_batch_size,
        grid_target,
        bias_target,
        std_target,
        calibration_min_idx,
        packet_size,
        iter_size,
        n_iter,
        prod,
        worker_id,
        defaults=defaults,
    )
    db.store.set_or_append("config", c.config_df)

    ########################################
    # Set up model, driver and grid
    ########################################

    # Why initialize the model here rather than relying on the user to pass an
    # already constructed model object?
    # 1. We can pass the seed here.
    # 2. We can pass the correct max_K and avoid a class of errors.
    model = modeltype(
        seed=c.model_seed,
        max_K=c.init_K * 2**c.n_K_double,
        **json.loads(c.model_kwargs),
    )
    null_hypos = _load_null_hypos(db) if g is None else g.null_hypos
    ada_driver = AdaCalibrationDriver(db, model, null_hypos, c)

    if g is not None:
        # Copy the input grid so that the caller is not surprised by any changes.
        g = copy.deepcopy(g)
        g.df["K"] = c.init_K

        # We process the tiles before adding them to the database so that the
        # database will be initialized with the correct set of columns.
        g_calibrated = ada_driver._process_tiles(g, 0)
        db.init_tiles(g_calibrated.df)
        _store_null_hypos(db, null_hypos)

    ########################################
    # Run adagrid!
    ########################################

    reports = []
    for ada_iter in range(c.n_iter):
        try:
            done, report = ada_driver.step(ada_iter)
            if callback is not None:
                callback(ada_iter, report, ada_driver)
            reports.append(report)
            if done:
                break
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            return ada_iter, reports, db

    return ada_iter + 1, reports, db


def verify_adagrid(df):
    inactive_ids = df.loc[~df["active"], "id"]
    assert inactive_ids.unique().shape == inactive_ids.shape

    parents = df["parent_id"].unique()
    parents_that_dont_exist = np.setdiff1d(parents, inactive_ids)
    inactive_tiles_with_no_children = np.setdiff1d(inactive_ids, parents)
    assert parents_that_dont_exist.shape[0] == 1
    assert parents_that_dont_exist[0] == 0
    assert inactive_tiles_with_no_children.shape[0] == 0
