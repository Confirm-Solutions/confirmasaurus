import logging
import time

import numpy as np
import pandas as pd

import imprint
from . import adagrid
from . import config

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

        rej_df = self.driver.validate(tiles_df, self.c["lam"], delta=self.delta)
        rej_df.insert(0, "processor_id", self.c["worker_id"])
        rej_df.insert(1, "processing_time", imprint.timer.simple_timer())
        rej_df.insert(2, "eligible", True)
        rej_df.insert(3, "grid_cost", rej_df["tie_bound"] - rej_df["tie_cp_bound"])
        rej_df.insert(4, "sim_cost", rej_df["tie_cp_bound"] - rej_df["tie_est"])
        rej_df.insert(5, "total_cost", rej_df["grid_cost"] + rej_df["sim_cost"])
        # TODO: think about convergence criterion and add orderer!!
        # It's possible to write a binary tile-wise criterion into orderer by
        # just setting it to 1 and 0.
        return pd.concat((tiles_df, rej_df), axis=1)

    def convergence_criterion(self, report):
        # max_tie_est = (
        #     self.db.con.execute("select max(tie_est) from tiles where active=true")
        #     .df()
        #     .iloc[0, 0]
        # )
        # work = self.db.con.execute(
        #     "select * from tiles"
        #     f"  where  active=true"
        #     f"         and (total_cost > {global_target}"
        #     f"              or (total_cost > {max_target}"
        #     f"                    and tie_bound > {max_tie_est}))"
        #     f" limit {packet_size}"
        # ).df()

        # step 2: check if there's anything left to do
        report["converged"] = False  # work.shape[0] == 0
        return report["converged"]

    def new_step(self, new_step_id, report):
        # TODO: need to replace the select_tiles call.
        # tiles = self.db.select_tiles(self.c["step_size"], "orderer")
        tiles = None
        logger.info(
            f"Preparing new step {new_step_id} with {tiles.shape[0]} parent tiles."
        )
        tiles["finisher_id"] = self.c["worker_id"]
        tiles["query_time"] = imprint.timer.simple_timer()
        if tiles.shape[0] == 0:
            return "empty"

        deepen_cheaper = tiles["sim_cost"] > tiles["grid_cost"]
        impossible = (tiles["K"] == max_K) & (
            (tiles["sim_cost"] > global_target)
            | ((tiles["sim_cost"] > max_target) & (tiles["tie_bound"] > max_tie_est))
        )
        tiles["deepen"] = deepen_cheaper & (tiles["K"] < max_K)
        tiles["refine"] = ~tiles["deepen"]
        tiles["refine"] &= ~impossible
        tiles["active"] = ~(tiles["refine"] | tiles["deepen"])


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
    max_target=0.001,
    global_target=0.002,
    n_steps: int = 100,
    step_size=2**10,
    n_iter=1000,
    packet_size: int = None,
    prod: bool = True,
    overrides: dict = None,
    callback=config.print_report,
):
    return adagrid.run(modeltype, g, db, locals(), "validation", callback)

    # # TODO: output how many tiles are left according to the criterion?
    # # TODO: order refinement and deepening by total_cost.
    # # TODO: move the query inside the database backend.

    # reports = []
    # ada_iter = 1
    # for ada_iter in range(1, n_iter):
    #     start_convergence = time.time()
    #     # step 1: grab a batch of the worst tiles.
    #     # TODO: move this into the DB interface.
    #     max_tie_est = (
    #         db.con.execute("select max(tie_est) from tiles where active=true")
    #         .df()
    #         .iloc[0, 0]
    #     )
    #     work = db.con.execute(
    #         "select * from tiles"
    #         f"  where  active=true"
    #         f"         and (total_cost > {global_target}"
    #         f"              or (total_cost > {max_target}"
    #         f"                    and tie_bound > {max_tie_est}))"
    #         f" limit {packet_size}"
    #     ).df()

    #     # step 2: check if there's anything left to do
    #     done = work.shape[0] == 0

    #     worst_tile = db.worst_tile("tie_bound")
    #     report = dict(
    #         i=ada_iter,
    #         n_work=work.shape[0],
    #         max_total_cost=work["total_cost"].max(),
    #         max_grid_cost=work["grid_cost"].max(),
    #         max_sim_cost=work["sim_cost"].max(),
    #         worst_tile_est=worst_tile["tie_est"].iloc[0],
    #         worst_tile_bound=worst_tile["tie_bound"].iloc[0],
    #         worst_tile_cost=worst_tile["total_cost"].iloc[0],
    #         runtime_convergence_check=time.time() - start_convergence,
    #     )

    #     if done:
    #         pprint(report)
    #         break

    #     # step 3: identify whether to refine or deepen
    #     start_refine_deepen = time.time()
    #     deepen_cheaper = work["sim_cost"] > work["grid_cost"]
    #     impossible = (work["K"] == max_K) & (
    #         (work["sim_cost"] > global_target)
    #         | ((work["sim_cost"] > max_target) & (work["tie_bound"] > max_tie_est))
    #     )
    #     work["deepen"] = deepen_cheaper & (work["K"] < max_K)
    #     work["refine"] = ~work["deepen"]
    #     work["refine"] &= ~impossible
    #     work["active"] = ~(work["refine"] | work["deepen"])

    #     # step 4: refine, deepen --> validate!
    #     n_refine = work["refine"].sum()
    #     n_deepen = work["deepen"].sum()
    #     report.update(
    #         dict(
    #             n_refine=n_refine,
    #             n_deepen=n_deepen,
    #             n_impossible=impossible.sum(),
    #         )
    #     )
    #     nothing_to_do = n_refine == 0 and n_deepen == 0
    #     if not nothing_to_do:
    #         g_new = refine_and_deepen(work, null_hypos)
    #         report["runtime_refine_deepen"] = time.time() - start_refine_deepen

    #         start_processing = time.time()
    #         g_val_new = _validation_process_tiles(
    #             db, ada_driver, g_new, lam, delta, ada_iter, transformation
    #         )
    #         db.write(g_val_new.df)
    #         report.update(
    #             dict(
    #                 n_processed=g_val_new.n_tiles,
    #                 K_distribution=g_val_new.df["K"].value_counts().to_dict(),
    #             )
    #         )

    #     db.finish(work)
    #     if not nothing_to_do:
    #         report["runtime_processing"] = time.time() - start_processing
    #     else:
    #         report["runtime_refine_deepen"] = time.time() - start_refine_deepen

    #     pprint(report)
    #     reports.append(report)
    # return ada_iter, reports, db
