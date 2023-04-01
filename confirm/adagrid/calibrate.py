"""

Broadly speaking, the adaptive calibration algorithm refines and deepens tiles until
the type I error, max(f_hat(lambda**)), is close to the type I error control
level, alpha.

Deciding to refine a tile is fairly easy because we can directly calculate the
type I error slack from continuous simulation extension: alpha - alpha0

On the other hand, deciding whether to deepen (that is, to increase the number
of simulations) is more complicated. The main tool is to calculate lambda* from
bootstrap resampling the test statistics. This gives us information on the
distribution of lambda* and allows estimating the bias and standard error of
the lambda* estimate. We call this the "tilewise bootstrap", often abbreviated
in the code to `twb`.

TODO: write more about the deepening process.
- we would like to inflate these minima.
- reference the paper on the issues of many near-critical tiles. minimum of gaussian
  draws as a useful model.


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
- alpha0: backward_cse(this_tile, alpha). Given global Type I Error control of
  alpha, we invert the CSE bounds to compute the maximum allowable Type I Error
  at the simulation point.
- grid_cost: alpha - alpha0
- impossible tiles are those for which K and alpha0 are not large enough to
  satisfy the calibration_min_idx constraint.
- orderer: the value by which we decide the next tiles to "process"
- processing tiles: deciding to refine or deepen and then simulating for those
  new tiles that resulted from refining or deepening.
- worst tile: the tile for which lams is smallest:
 `lams[worst_tile] == lamss == lambda**`
"""
import logging

import numpy as np
import pandas as pd

from . import bootstrap
from .adagrid import pass_control_to_backend
from .adagrid import print_report
from .const import MAX_STEP

logger = logging.getLogger(__name__)


class AdaCalibrate:
    def __init__(self, model, null_hypos, db, cfg, callback):
        self.null_hypos = null_hypos
        self.db = db
        self.cfg = cfg
        self.callback = callback

        Ks = self.cfg["init_K"] * 2 ** np.arange(self.cfg["n_K_double"] + 1)
        self.max_K = Ks[-1]
        self.driver = bootstrap.BootstrapCalibrate(
            model, self.cfg["bootstrap_seed"], self.cfg["nB"], Ks
        )

    def get_orderer(self):
        return "orderer"

    @staticmethod
    def get_columns(self):
        return (
            [
                "orderer",
                "idx",
                "alpha0",
                "impossible",
                "twb_max_lams",
                "twb_mean_lams",
                "twb_min_lams",
                "lams",
            ]
            + [f"B_lams{i}" for i in range(self.cfg["nB"])]
            + [f"twb_lams{i}" for i in range(self.cfg["nB"])]
        )

    def process_tiles(self, *, tiles_df, tile_batch_size):
        # This method actually runs the calibration and bootstrapping.
        # It is called once per iteration.
        # Several auxiliary fields are calculated because they are needed for
        # selecting the next iteration's tiles: impossible and orderer

        lams_df = self.driver.bootstrap_calibrate(
            tiles_df,
            self.cfg["alpha"],
            calibration_min_idx=self.cfg["calibration_min_idx"],
            tile_batch_size=tile_batch_size,
        )
        lams_df.insert(0, "completion_step", MAX_STEP)
        lams_df.insert(
            1,
            "orderer",
            # Where calibration is impossible due to either small K or small alpha0,
            # the orderer is set to -inf so that such tiles are guaranteed to
            # be processed.
            np.minimum(
                lams_df["twb_min_lams"],
                np.where(lams_df["impossible"], -np.inf, np.inf),
            ),
        )
        return pd.concat((tiles_df, lams_df), axis=1)

    def convergence_criterion(self, basal_step_id):
        ########################################
        # Step 2: Convergence criterion! In terms of:
        # - bias
        # - standard deviation
        # - grid cost (i.e. alpha - alpha0)
        #
        # The bias and standard deviation are calculated using the bootstrap.
        ########################################
        worst_tile_impossible = self.db.worst_tile(basal_step_id, "impossible")

        # If there are no tiles, we are done.
        if worst_tile_impossible.shape[0] == 0:
            return True, None, {}

        any_impossible = worst_tile_impossible["impossible"].iloc[0]
        if any_impossible:
            return False, None, {}

        worst_tile_df = self.db.worst_tile(basal_step_id, "lams")
        worst_tile = worst_tile_df.iloc[0]
        lamss = worst_tile["lams"]

        # We determine the bias by comparing the Type I error at the worst
        # tile for each lambda**_B:
        logger.debug("Computing bias and standard deviation of lambda**")
        B_lamss = self.db.bootstrap_lamss(basal_step_id)
        worst_tile_tie_sum = self.driver.many_rej(
            worst_tile_df, np.array([lamss] + list(B_lamss))
        ).iloc[0]
        worst_tile_tie_est = worst_tile_tie_sum / worst_tile["K"]

        # Given these TIE values, we can compute bias, standard deviation and
        # spread.
        bias_tie = worst_tile_tie_est[0] - worst_tile_tie_est[1:].mean()
        std_tie = worst_tile_tie_est.std()
        spread_tie = worst_tile_tie_est.max() - worst_tile_tie_est.min()
        grid_cost = self.cfg["alpha"] - worst_tile["alpha0"]

        report = dict(
            bias_tie=bias_tie,
            std_tie=std_tie,
            spread_tie=spread_tie,
            grid_cost=grid_cost,
            lamss=lamss,
        )
        report["min(B_lamss)"] = min(B_lamss)
        report["max(B_lamss)"] = max(B_lamss)
        report["tie_{k}(lamss)"] = worst_tile_tie_est[0]
        report["tie + slack"] = worst_tile_tie_est[0] + grid_cost + bias_tie

        # The convergence criterion itself.
        report["converged"] = (
            (bias_tie < self.cfg["bias_target"])
            and (std_tie < self.cfg["std_target"])
            and (grid_cost < self.cfg["grid_target"])
        )
        return report["converged"], None, report

    def select_tiles(self, basal_step_id, new_step_id, _):
        tiles_df = self.db.next(
            basal_step_id, new_step_id, self.cfg["step_size"], "orderer"
        )
        logger.info(f"Preparing new step with {tiles_df.shape[0]} parent tiles.")
        if tiles_df.shape[0] == 0:
            return None, {}

        ########################################
        # Is deepening likely to be enough?
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
        # lambda* of the tile with the lowest mean lambda* with two modifications:
        # - We increase to the maximum K allowed.
        # - We shrink the radius of the tile so that alpha0 \approx alpha.
        #
        # If the tile's min lambda* is less the mean lambda* of this worst
        # tile, then the tile actually has a chance of being the worst tile. In
        # which case, we choose the more expensive option of refining the tile.

        twb_worst_tile = self.db.worst_tile(basal_step_id, "twb_mean_lams")
        for col in twb_worst_tile.columns:
            if col.startswith("radii"):
                twb_worst_tile[col] = 1e-6
        twb_worst_tile["K"] = self.max_K
        twb_worst_tile_lams = self.driver.bootstrap_calibrate(
            twb_worst_tile,
            self.cfg["alpha"],
            calibration_min_idx=self.cfg["calibration_min_idx"],
            tile_batch_size=1,
        )
        twb_worst_tile_mean_lams = twb_worst_tile_lams["twb_mean_lams"].iloc[0]
        deepen_likely_to_work = tiles_df["twb_mean_lams"] > twb_worst_tile_mean_lams

        ########################################
        # Decide whether to refine or deepen
        # THIS IS IT! This is where we decide whether to refine or deepen.
        #
        # The decision criteria is best described by the code below.
        ########################################
        at_max_K = tiles_df["K"] == self.max_K

        grid_cost = self.cfg["alpha"] - tiles_df["alpha0"]
        tiles_df["refine"] = (grid_cost > self.cfg["grid_target"]) & (
            (~deepen_likely_to_work) | at_max_K
        )
        tiles_df["deepen"] = (~tiles_df["refine"]) & (~at_max_K)

        report = dict(n_impossible=tiles_df["impossible"].sum())
        return tiles_df, report


def ada_calibrate(
    model_type,
    *,
    g=None,
    model_seed: int = 0,
    model_kwargs: dict = None,
    alpha: float = 0.025,
    init_K: int = 2**13,
    n_K_double: int = 4,
    bootstrap_seed: int = 0,
    nB: int = 50,
    tile_batch_size: int = None,
    grid_target: float = 0.001,
    bias_target: float = 0.001,
    std_target: float = 0.002,
    calibration_min_idx: int = 40,
    n_steps: int = 100,
    timeout: int = 60 * 60 * 12,
    step_size: int = 2**10,
    packet_size: int = None,
    n_parallel_steps: int = 1,
    record_system: bool = True,
    clickhouse_service: str = None,
    job_name_prefix: str = None,
    job_name: str = None,
    overrides: dict = None,
    callback=print_report,
    backend=None,
):
    """
    The main entrypoint for the adaptive calibration algorithm.

    Args:
        model_type: The model class to use.
        g: The initial grid.
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
        calibration_min_idx: The minimum calibration selection index. We enforce that:
            `alpha0 >= (calibration_min_idx + 1) / (K + 1)`
            A larger value will reduce the variance of lambda* but
            will require more computational effort because K and/or
            alpha0 will need to be larger. Defaults to 40.
        n_steps: The number of Adagrid steps to run. Defaults to 100.
        timeout: The maximum number of seconds to run for. Defaults to 12 hours.
        step_size: The number of tiles in an Adagrid step produced by a single
           Adagrid tile selection step. This is different from
           packet_size because we select tiles once and then run many
           simulation "iterations" in parallel each processing one
           packet of tiles. Defaults to 2**10.
        packet_size: The number of tiles to process per iteration. Defaults to
            None. If None, we use the same value as step_size.
        n_parallel_steps: The number of Adagrid steps to run in parallel.
            Setting this parameter to anything greater than 1 will cause the steps
            to be based on lagged data. For example, with n_parallel_steps=2,
            step K will be based on data from step K-2. Defaults to 1.
        record_system: If True, we will collection extra system
            configuration info. Setting this to False will make startup time
            a bit faster.
        clickhouse_service: If 'PROD', we mirror all database inserts to a the
            prod Clickhouse service. If 'TEST' we mirror to the test service. If
            None, we do not mirror inserts. Default is None.
        job_name_prefix: A prefix to use for the job name. This is useful for
            grouping jobs together. Defaults to None. If job_name is set, this is
            ignored. If job_name is not set, this is used to generate a job
            name like: `{job_name_prefix}_{time.strftime("%Y%m%d_%H%M%S")}`.
        job_name: The job name is used for the database file used by DuckDB and
            for storing long-term backups in Clickhouse. By default (None), an
            in-memory DuckDB is used and unnamed_YYYYMMDD_HHMMSS is chosen for
            Clickhouse.
        overrides: If this call represents a continuation of an existing
            adagrid job, the overrides dictionary will be used to override the
            preset configuration settings. All other arguments will be ignored.
            If this calls represents a new adagrid job, this argument is
            ignored.
        callback: A function accepting three arguments (report, db)
            that can perform some reporting or printing at each iteration.
            Defaults to print_report.
        backend: The backend to use for running the job. Defaults to running
            locally.

    Returns:
        ada_iter: The final iteration number.
        reports: A list of the report dicts from each iteration.
        db: The database object used for the run. This can be used to
            inspect the results of the run.
    """
    return pass_control_to_backend(AdaCalibrate, locals())
