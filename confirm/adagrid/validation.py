import copy
import time
from pprint import pprint

import pandas as pd

from .calibration import refine_deepen
from .db import DuckDBTiles
from imprint import driver


def _validation_process_tiles(db, driver, g, lam, delta, i, transformation):
    print("processing ", g.n_tiles)
    if transformation is None:
        computational_df = g.df
    else:
        theta, radii, null_truth = transformation(
            g.get_theta(), g.get_radii(), g.get_null_truth()
        )
        d = theta.shape[1]
        indict = {}
        indict["K"] = g.df["K"]
        for i in range(d):
            indict[f"theta{i}"] = theta[:, i]
        for i in range(d):
            indict[f"radii{i}"] = radii[:, i]
        for j in range(null_truth.shape[1]):
            indict[f"null_truth{j}"] = null_truth[:, j]
        computational_df = pd.DataFrame(indict)

    rej_df = driver.validate(computational_df, lam, delta=delta)
    rej_df["grid_cost"] = rej_df["tie_bound"] - rej_df["tie_cp_bound"]
    rej_df["sim_cost"] = rej_df["tie_cp_bound"] - rej_df["tie_est"]
    rej_df["total_cost"] = rej_df["grid_cost"] + rej_df["sim_cost"]

    g_val = g.add_cols(rej_df)
    g_val.df["worker_id"] = db.worker_id
    g_val.df["birthiter"] = i
    g_val.df["birthtime"] = time.time()
    return g_val


def ada_validate(
    modeltype,
    *,
    g,
    lam,
    db=None,
    transformation=None,
    delta=0.01,
    model_seed=0,
    init_K=2**13,
    n_K_double=4,
    tile_batch_size=64,
    max_target=0.001,
    global_target=0.002,
    # grid_target=None, # might be a nice feature?
    # sim_target=None, # might be a nice feature?
    iter_size=2**10,
    n_iter=1000,
    model_kwargs=None,
):
    # TODO: output how many tiles are left according to the criterion?
    # TODO: order refinement and deepening by total_cost.
    # TODO: clean up...
    # TODO: move the query inside the database backend.
    if model_kwargs is None:
        model_kwargs = {}
    max_K = init_K * 2**n_K_double
    model = modeltype(seed=model_seed, max_K=max_K, **model_kwargs)
    ada_driver = driver.Driver(model, tile_batch_size=tile_batch_size)

    if db is None:
        if g is None:
            raise ValueError(
                "Must provide either an initial grid or an existing"
                " database! Set either g or db."
            )
        db = DuckDBTiles.connect()

        # TODO: fix this, not right in the midterm for restarts
        g = copy.deepcopy(g)
        null_hypos = g.null_hypos
        g.df["K"] = init_K
        g_val = _validation_process_tiles(
            db, ada_driver, g, lam, delta, 0, transformation
        )
        db.init_tiles(g_val.df)
    else:
        db = db
        null_hypos = g.null_hypos

    reports = []
    ada_iter = 1
    for ada_iter in range(1, n_iter):
        start_convergence = time.time()
        # step 1: grab a batch of the worst tiles.
        # TODO: move this into the DB interface.
        max_tie_est = (
            db.con.execute("select max(tie_est) from tiles where active=true")
            .df()
            .iloc[0, 0]
        )
        work = db.con.execute(
            "select * from tiles"
            f"  where  active=true"
            f"         and (total_cost > {global_target}"
            f"              or (total_cost > {max_target}"
            f"                    and tie_bound > {max_tie_est}))"
            f" limit {iter_size}"
        ).df()

        # step 2: check if there's anything left to do
        done = work.shape[0] == 0

        worst_tile = db.worst_tile("tie_bound")
        report = dict(
            i=ada_iter,
            n_work=work.shape[0],
            max_total_cost=work["total_cost"].max(),
            max_grid_cost=work["grid_cost"].max(),
            max_sim_cost=work["sim_cost"].max(),
            worst_tile_est=worst_tile["tie_est"].iloc[0],
            worst_tile_bound=worst_tile["tie_bound"].iloc[0],
            worst_tile_cost=worst_tile["total_cost"].iloc[0],
            runtime_convergence_check=time.time() - start_convergence,
        )

        if done:
            pprint(report)
            break

        # step 3: identify whether to refine or deepen
        start_refine_deepen = time.time()
        deepen_cheaper = work["sim_cost"] > work["grid_cost"]
        impossible = (work["K"] == max_K) & (
            (work["sim_cost"] > global_target)
            | ((work["sim_cost"] > max_target) & (work["tie_bound"] > max_tie_est))
        )
        work["deepen"] = deepen_cheaper & (work["K"] < max_K)
        work["refine"] = ~work["deepen"]
        work["refine"] &= ~impossible
        work["active"] = ~(work["refine"] | work["deepen"])

        # step 4: refine, deepen --> validate!
        n_refine = work["refine"].sum()
        n_deepen = work["deepen"].sum()
        report.update(
            dict(
                n_refine=n_refine,
                n_deepen=n_deepen,
                n_impossible=impossible.sum(),
            )
        )
        nothing_to_do = n_refine == 0 and n_deepen == 0
        if not nothing_to_do:
            g_new = refine_deepen(work, null_hypos)
            report["runtime_refine_deepen"] = time.time() - start_refine_deepen

            start_processing = time.time()
            g_val_new = _validation_process_tiles(
                db, ada_driver, g_new, lam, delta, ada_iter, transformation
            )
            db.write(g_val_new.df)
            report.update(
                dict(
                    n_processed=g_val_new.n_tiles,
                    K_distribution=g_val_new.df["K"].value_counts().to_dict(),
                )
            )

        db.finish(work)
        if not nothing_to_do:
            report["runtime_processing"] = time.time() - start_processing
        else:
            report["runtime_refine_deepen"] = time.time() - start_refine_deepen

        pprint(report)
        reports.append(report)
    return ada_iter, reports, db
