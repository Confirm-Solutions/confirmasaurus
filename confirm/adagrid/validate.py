import numpy as np
import pandas as pd

import imprint
from . import adagrid

logger = imprint.log.getLogger(__name__)


class AdaValidate:
    def __init__(self, db, model, null_hypos, c):
        self.db = db
        self.model = model
        self.null_hypos = null_hypos
        self.c = c

        self.Ks = self.c["init_K"] * 2 ** np.arange(self.c["n_K_double"] + 1)
        self.max_K = self.Ks[-1]
        self.driver = imprint.driver.Driver(
            self.model, tile_batch_size=self.c["tile_batch_size"]
        )

    def get_orderer(self):
        return "total_cost_order, tie_bound_order"

    def process_tiles(self, *, tiles_df, report):
        # TODO: bring back transformations?? in a more general way?
        # if transformation is None:
        #     computational_df = g.df
        # else:
        #     theta, radii, null_truth = transformation(
        #         g.get_theta(), g.get_radii(), g.get_null_truth()
        #     )
        #     d = theta.shape[1]
        #     indict = {}
        #     indict["K"] = g.df["K"]
        #     for i in range(d):
        #         indict[f"theta{i}"] = theta[:, i]
        #     for i in range(d):
        #         indict[f"radii{i}"] = radii[:, i]
        #     for j in range(null_truth.shape[1]):
        #         indict[f"null_truth{j}"] = null_truth[:, j]
        #     computational_df = pd.DataFrame(indict)

        rej_df = self.driver.validate(tiles_df, self.c["lam"], delta=self.c["delta"])
        rej_df.insert(0, "processor_id", self.c["worker_id"])
        rej_df.insert(1, "processing_time", imprint.timer.simple_timer())
        rej_df.insert(2, "eligible", True)
        rej_df.insert(3, "grid_cost", rej_df["tie_bound"] - rej_df["tie_cp_bound"])
        rej_df.insert(4, "sim_cost", rej_df["tie_cp_bound"] - rej_df["tie_est"])
        rej_df.insert(5, "total_cost", rej_df["grid_cost"] + rej_df["sim_cost"])

        # The orderer for validation consists of a tuple
        # total_cost_order: the first entry is the total_cost thresholded by
        #   whether the total_cost is greater than the convergence criterion.
        # tie_bound_order: the second entry is the tie_bound thresholded by
        #   whether the cost is greater than max_target
        # This means that we'll first handle tiles for which the total_cost is
        # too high. Then, after we've exhausted those tiles, we'll handle tiles
        # where the total_cost is low enough but the tie_bound is high enough
        # that we want to use the tighter convergence criterion from
        # max_target.
        #
        # This allows for having a global acceptable slack with global_target
        # and then a separate tighter criterion near the maximum tie_bound.
        rej_df.insert(
            6,
            "total_cost_order",
            -rej_df["total_cost"] * (rej_df["total_cost"] > self.c["global_target"]),
        )
        rej_df.insert(
            7,
            "tie_bound_order",
            -rej_df["tie_bound"] * (rej_df["total_cost"] > self.c["max_target"]),
        )
        return pd.concat((tiles_df.drop("K", axis=1), rej_df), axis=1)

    def convergence_criterion(self, worker_id, report):
        max_tie_est = self.db.worst_tile(worker_id, "tie_est desc")["tie_est"].iloc[0]
        next_tile = self.db.worst_tile(
            worker_id, "total_cost_order, tie_bound_order"
        ).iloc[0]
        report["converged"] = self._are_tiles_done(next_tile, max_tie_est)
        report.update(
            dict(
                max_tie_est=max_tie_est,
                next_tile_tie_est=next_tile["tie_est"],
                next_tile_tie_bound=next_tile["tie_bound"],
                next_tile_sim_cost=next_tile["sim_cost"],
                next_tile_grid_cost=next_tile["grid_cost"],
                next_tile_total_cost=next_tile["total_cost"],
                next_tile_K=next_tile["K"],
                next_tile_at_max_K=next_tile["K"] == self.max_K,
            )
        )
        return report["converged"], max_tie_est

    def _are_tiles_done(self, tiles, max_tie_est):
        return ~(
            (tiles["total_cost_order"] < 0)
            | (((tiles["tie_bound_order"] < 0) & (tiles["tie_bound"] > max_tie_est)))
        )

    def select_tiles(self, coordination_id, report, max_tie_est):
        # TODO: output how many tiles are left according to the criterion?
        raw_tiles = self.db.next(
            coordination_id,
            self.c["worker_id"],
            self.c["step_size"],
            "total_cost_order, tie_bound_order",
        )
        include = ~self._are_tiles_done(raw_tiles, max_tie_est)
        tiles_df = raw_tiles[include].copy()
        logger.info(f"Preparing new step with {tiles_df.shape[0]} parent tiles.")
        if tiles_df.shape[0] == 0:
            return None

        report.update(
            dict(
                n_tiles=tiles_df.shape[0],
                step_max_total_cost=tiles_df["total_cost"].max(),
                step_max_grid_cost=tiles_df["grid_cost"].max(),
                step_max_sim_cost=tiles_df["sim_cost"].max(),
            )
        )

        deepen_cheaper = tiles_df["sim_cost"] > tiles_df["grid_cost"]
        needs_refine = (tiles_df["grid_cost"] > self.c["global_target"]) | (
            (tiles_df["grid_cost"] > self.c["max_target"])
            & (tiles_df["tie_bound"] > max_tie_est)
        )
        tiles_df["deepen"] = deepen_cheaper & (tiles_df["K"] < self.max_K)
        tiles_df["refine"] = (~tiles_df["deepen"]) & needs_refine
        tiles_df["refine"] |= ((~tiles_df["refine"]) & (~tiles_df["deepen"])) & (
            tiles_df["grid_cost"] > (tiles_df["sim_cost"] / 5)
        )
        return tiles_df


def ada_validate(
    model_type,
    *,
    lam,
    g=None,
    db=None,
    model_seed=0,
    model_kwargs=None,
    delta=0.01,
    init_K=2**13,
    n_K_double=4,
    tile_batch_size=64,
    max_target=0.002,
    global_target=0.005,
    n_steps: int = 100,
    step_size=2**10,
    n_iter=1000,
    packet_size: int = None,
    prod: bool = True,
    overrides: dict = None,
    callback=adagrid.print_report,
    backend=adagrid.LocalBackend(),
):
    """
    The entrypoint for the adaptive validation algorithm.

    Args:
        model_type: The model class to use.
        lam: The test statistic threshold to use for deciding to reject the
            null hypothesis.
        g: The initial grid. If not provided, the grid is assumed to be stored
            in the database. At least one of `g` or `db` must be provided.
        db: The database backend to use. Defaults to `db.DuckDB.connect()`.
        model_seed: The random seed for the model. Defaults to 0.
        model_kwargs: Additional keyword arguments for constructing the Model
            object. Defaults to None.
        delta: The pointwise Clopper-Pearson confidence intervals will have a
            fractional width of 1 - `delta`. Defaults to 0.01 which results in a
            99% confidence interval.
        init_K: Initial K for the first tiles. Defaults to 2**13.
        n_K_double: The number of doublings of K. The maximum K will be
            `init_K * 2 ** (n_K_double + 1)`. Defaults to 4.
        tile_batch_size: The number of tiles to simulate in a single batch.
            Defaults to 64 on GPU and 4 on CPU.
        max_target: The limit on allowed slack in fraction Type I Error for
            tiles that have a Type I Error bound that is above the worst case
            simulated Type I Error. This convergence criterion parameter is
            useful for tightening the bound in the areas where it matters most.
            Defaults to 0.002.
        global_target: The limit on allowed slack in fraction Type I Error for
            any tile regardless of its Type I Error bound value. This convergence
            criterion parameter is useful for getting a tighter fit to the Type I
            Error surface throughout parameter space regardless of the value of the
            Type I Error. Defaults to 0.005.
        n_steps: The number of Adagrid steps to run. Defaults to 100.
        step_size: The number of tiles in an Adagrid step produced by a single
           Adagrid tile selection step. This is different from
           packet_size because we select tiles once and then run many
           simulation "iterations" in parallel each processing one
           packet of tiles. Defaults to 2**10.
        n_iter: The number of packets this worker should simulate before
            exiting. Defaults to None which places no limit. Limiting the number of
            packets is useful for stopping a worker after a specified amount of
            work.
        packet_size: The number of tiles to process per iteration. Defaults to
            None. If None, we use the same value as step_size.
        prod: Is this a production run? If so, we will collection extra system
            configuration info. Setting this to False will make startup time
            a bit faster. Defaults to True.
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
    return adagrid.run(AdaValidate, **locals())
