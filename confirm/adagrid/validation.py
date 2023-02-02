import logging

import numpy as np
import pandas as pd

import imprint
from . import adagrid

logger = logging.getLogger(__name__)


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
        return pd.concat((tiles_df, rej_df), axis=1)

    def convergence_criterion(self, report):
        max_tie_est = self.db.worst_tile("tie_est desc")["tie_est"].iloc[0]
        next_tile = self.db.worst_tile("total_cost_order, tie_bound_order").iloc[0]
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

    def new_step(self, new_step_id, report, max_tie_est):
        # TODO: output how many tiles are left according to the criterion?
        # TODO: lots of new_step is duplicated with calibration.
        raw_tiles = self.db.select_tiles(
            self.c["step_size"], "total_cost_order, tie_bound_order"
        )
        include = ~self._are_tiles_done(raw_tiles, max_tie_est)
        tiles = raw_tiles[include].copy()
        logger.info(
            f"Preparing new step {new_step_id} with {tiles.shape[0]} parent tiles."
        )
        tiles["finisher_id"] = self.c["worker_id"]
        tiles["query_time"] = imprint.timer.simple_timer()
        if tiles.shape[0] == 0:
            return "empty"

        report.update(
            dict(
                n_tiles=tiles.shape[0],
                step_max_total_cost=tiles["total_cost"].max(),
                step_max_grid_cost=tiles["grid_cost"].max(),
                step_max_sim_cost=tiles["sim_cost"].max(),
            )
        )

        deepen_cheaper = tiles["sim_cost"] > tiles["grid_cost"]
        needs_refine = (tiles["grid_cost"] > self.c["global_target"]) | (
            (tiles["grid_cost"] > self.c["max_target"])
            & (tiles["tie_bound"] > max_tie_est)
        )
        tiles["deepen"] = deepen_cheaper & (tiles["K"] < self.max_K)
        tiles["refine"] = (~tiles["deepen"]) & needs_refine
        tiles["refine"] |= ((~tiles["refine"]) & (~tiles["deepen"])) & (
            tiles["grid_cost"] > (tiles["sim_cost"] / 5)
        )
        tiles["active"] = ~(tiles["refine"] | tiles["deepen"])

        # Record what we decided to do.
        self.db.finish(
            tiles[
                [
                    "id",
                    "step_id",
                    "step_iter",
                    "active",
                    "query_time",
                    "finisher_id",
                    "refine",
                    "deepen",
                ]
            ]
        )

        n_refine = tiles["refine"].sum()
        n_deepen = tiles["deepen"].sum()
        report.update(
            dict(
                n_refine=n_refine,
                n_deepen=n_deepen,
                n_complete=tiles["active"].sum(),
            )
        )

        ########################################
        # Step 6: Deepen and refine tiles.
        ########################################
        nothing_to_do = n_refine == 0 and n_deepen == 0
        if nothing_to_do:
            return "empty"

        df = adagrid.refine_and_deepen(
            tiles, self.null_hypos, self.max_K, self.c["worker_id"]
        ).df
        df["step_id"] = new_step_id
        df["step_iter"], n_packets = adagrid.step_iter_assignments(
            df, self.c["packet_size"]
        )
        df["creator_id"] = self.c["worker_id"]
        df["creation_time"] = imprint.timer.simple_timer()

        n_tiles = df.shape[0]
        logger.debug(
            f"new step {(new_step_id, 0, n_packets, n_tiles)} "
            f"n_tiles={n_tiles} packet_size={self.c['packet_size']}"
        )
        self.db.set_step_info(
            step_id=new_step_id, step_iter=0, n_iter=n_packets, n_tiles=n_tiles
        )

        self.db.insert_tiles(df)
        report.update(
            dict(
                n_new_tiles=n_tiles, new_K_distribution=df["K"].value_counts().to_dict()
            )
        )
        return new_step_id


def ada_validate(
    modeltype,
    lam,
    *,
    g=None,
    db=None,
    transformation=None,
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
):
    return adagrid.run(modeltype, g, db, locals(), AdaValidate, callback)
