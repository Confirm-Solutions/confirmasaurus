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


TODO: Code snippet for running bootstrap calibration on a particular grid. Left this
here because this is not as easy as it should be.
```
from confirm.adagrid.calibration import AdaCalibrationDriver, CalibrationConfig
import json
gtemp = ip.Grid(db1.get_all())
null_hypos = [ip.hypo("x0 < 0")]
c= CalibrationConfig(
    ZTest1D,
    *[None] * 16,
    defaults=db1.store.get('config').iloc[0].to_dict()
)
model = ZTest1D(
    seed=c['model_seed'],
    max_K=c['init_K'] * 2**c['n_K_double'],
    **json.loads(c['model_kwargs']),
)
driver = AdaCalibrationDriver(None, model, null_hypos, c)
driver.bootstrap_calibrate(gtemp.df, 0.025)
gtemp.df['K'].value_counts()
```

"""
import copy
import json
import time
import warnings

import numpy as np
import pandas as pd

import imprint.log
from . import bootstrap
from . import config
from . import distributed
from .db import DuckDBTiles
from imprint import grid
from imprint.timer import simple_timer

logger = imprint.log.getLogger(__name__)


class AdaCalibration:
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

    def __init__(self, db, model, null_hypos, c):
        self.db = db
        self.model = model
        self.null_hypos = null_hypos
        self.c = c

        self.Ks = self.c["init_K"] * 2 ** np.arange(self.c["n_K_double"] + 1)
        self.max_K = self.Ks[-1]
        self.driver = bootstrap.BootstrapCalibrate(
            model,
            self.c["bootstrap_seed"],
            self.c["nB"],
            self.Ks,
            tile_batch_size=self.c["tile_batch_size"],
            worker_id=self.c["worker_id"],
        )

    def process_tiles(self, *, tiles_df, report):
        # This method actually runs the calibration and bootstrapping.
        # It is called once per iteration.
        # Several auxiliary fields are calculated because they are needed for
        # selecting the next iteration's tiles: impossible and orderer

        lams_df = self.driver.bootstrap_calibrate(tiles_df, self.c["alpha"])
        lams_df.insert(0, "processor_id", self.c["worker_id"])
        lams_df.insert(1, "processing_time", simple_timer())
        lams_df.insert(2, "eligible", True)

        # we use insert here to order columns nicely for reading raw data
        lams_df.insert(3, "grid_cost", self.c["alpha"] - lams_df["alpha0"])
        lams_df.insert(
            4,
            "impossible",
            lams_df["alpha0"]
            < (self.c["calibration_min_idx"] + 1) / (tiles_df["K"] + 1),
        )

        lams_df.insert(
            5,
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

    def convergence_criterion(self, report):
        ########################################
        # Step 2: Convergence criterion! In terms of:
        # - bias
        # - standard deviation
        # - grid cost (i.e. alpha - alpha0)
        #
        # The bias and standard deviation are calculated using the bootstrap.
        ########################################
        any_impossible = self.db.worst_tile("impossible")["impossible"].iloc[0]
        if any_impossible:
            return False

        worst_tile = self.db.worst_tile("lams")
        lamss = worst_tile["lams"].iloc[0]

        # We determine the bias by comparing the Type I error at the worst
        # tile for each lambda**_B:
        B_lamss = self.db.bootstrap_lamss()
        worst_tile_tie_sum = self.driver.many_rej(
            worst_tile, np.array([lamss] + list(B_lamss))
        ).iloc[0]
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
            (bias_tie < self.c["bias_target"])
            and (std_tie < self.c["std_target"])
            and (grid_cost < self.c["grid_target"])
        )
        return report["converged"]

    def new_step(self, new_step_id, report):
        tiles = self.db.select_tiles(self.c["step_size"], "orderer")
        logger.info(
            f"Preparing new step {new_step_id} with {tiles.shape[0]} parent tiles."
        )
        tiles["finisher_id"] = self.c["worker_id"]
        tiles["query_time"] = simple_timer()
        if tiles.shape[0] == 0:
            return "empty"

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
        twb_worst_tile = self.db.worst_tile("twb_mean_lams")
        for col in twb_worst_tile.columns:
            if col.startswith("radii"):
                twb_worst_tile[col] = 1e-6
        twb_worst_tile["K"] = self.max_K
        twb_worst_tile_lams = self.driver.bootstrap_calibrate(
            twb_worst_tile, self.c["alpha"], tile_batch_size=1
        )
        twb_worst_tile_mean_lams = twb_worst_tile_lams["twb_mean_lams"].iloc[0]
        deepen_likely_to_work = tiles["twb_mean_lams"] > twb_worst_tile_mean_lams

        ########################################
        # Step 5: Decide whether to refine or deepen
        # THIS IS IT! This is where we decide whether to refine or deepen.
        #
        # The decision criteria is best described by the code below.
        ########################################
        at_max_K = tiles["K"] == self.max_K
        tiles["refine"] = (tiles["grid_cost"] > self.c["grid_target"]) & (
            (~deepen_likely_to_work) | at_max_K
        )
        tiles["deepen"] = (~tiles["refine"]) & (~at_max_K)
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
                n_impossible=tiles["impossible"].sum(),
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

        df = refine_deepen(tiles, self.null_hypos, self.max_K, self.c["worker_id"]).df
        df["step_id"] = new_step_id
        df["step_iter"], n_packets = step_iter_assignments(df, self.c["packet_size"])
        df["creator_id"] = self.c["worker_id"]
        df["creation_time"] = simple_timer()

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


def step_iter_assignments(df, packet_size):
    # Randomly assign tiles to packets.
    # TODO: this could be improved to better balance the load.
    # There are two load balancing concerns:
    # - a packet with tiles with a given K value should have lots of
    #   that K so that we have enough work to saturate GPU threads.
    # - we should divide the work evenly among the workers.
    n_tiles = df.shape[0]
    n_packets = int(np.ceil(n_tiles / packet_size))
    splits = np.array_split(np.arange(n_tiles), n_packets)
    assignment = np.empty(n_tiles, dtype=np.int32)
    for i in range(n_packets):
        assignment[splits[i]] = i
    rng = np.random.default_rng()
    rng.shuffle(assignment)
    return assignment, n_packets


# TODO: rename
def refine_deepen(g, null_hypos, max_K, worker_id):
    g_deepen_in = grid.Grid(g.loc[g["deepen"] & (g["K"] < max_K)], worker_id)
    g_deepen = grid.init_grid(
        g_deepen_in.get_theta(),
        g_deepen_in.get_radii(),
        worker_id=worker_id,
        parents=g_deepen_in.df["id"],
    )

    # We just multiply K by 2 to deepen.
    # TODO: it's possible to do better by multiplying by 4 or 8
    # sometimes when a tile clearly needs *way* more sims. how to
    # determine this?
    g_deepen.df["K"] = g_deepen_in.df["K"] * 2

    g_refine_in = grid.Grid(g.loc[g["refine"]], worker_id)
    inherit_cols = ["K"]
    # TODO: it's possible to do better by refining by more than just a
    # factor of 2.
    g_refine = g_refine_in.refine(inherit_cols)

    ########################################
    # Step 7: Simulate the new tiles and write to the DB.
    ########################################
    return g_refine.concat(g_deepen).add_null_hypos(null_hypos, inherit_cols).prune()


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
    step_size: int = 2**10,
    n_iter: int = 100,
    packet_size: int = None,
    prod: bool = True,
    overrides: dict = None,
    callback=config.print_report,
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
        calibration_min_idx: The minimum calibration selection index. We enforce that:
                        `alpha0 >= (calibration_min_idx + 1) / (K + 1)`
                        A larger value will reduce the variance of lambda* but
                        will require more computational effort because K and/or
                        alpha0 will need to be larger. Defaults to 40.
        n_steps: The number of Adagrid steps to run. Defaults to 100.
        step_size: The number of tiles in an Adagrid step produced by a single
                   Adagrid tile selection step. This is different from
                   packet_size because we select tiles once and then run many
                   simulation "iterations" in parallel to process those
                   tiles. Defaults to 2**10.
        n_iter: The number of packets to simulate. Defaults to None which
                places no limit. Limiting the number of packets is useful for
                stopping a worker after a specified amount of work.
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
        callback: A function accepting three arguments (iter, report, db)
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

    is_continuation = g is None
    worker_id = db.new_worker()
    imprint.log.worker_id.set(worker_id)

    ########################################
    # Store config
    ########################################

    model_name = modeltype.__name__
    if is_continuation:
        cfg_dict = db.store.get("config").iloc[0].to_dict()
        overrides = overrides or {}
        # Some parameters cannot be overridden because the job just wouldn't
        # make sense anymore.
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
            if k in overrides:
                raise ValueError(f"Parameter {k} cannot be overridden.")
    else:
        cfg_dict = locals()
        if overrides is not None:
            warnings.warn("Overrides are ignored when starting a new job.")
        overrides = {}
    c = config.prepare_config(cfg_dict, overrides, worker_id, prod)
    db.store.set_or_append("config", pd.DataFrame([c]))

    ########################################
    # Set up model, driver and grid
    ########################################

    # Why initialize the model here rather than relying on the user to pass an
    # already constructed model object?
    # 1. We can pass the seed here.
    # 2. We can pass the correct max_K and avoid a class of errors.
    model = modeltype(
        seed=c["model_seed"],
        max_K=c["init_K"] * 2 ** c["n_K_double"],
        **json.loads(c["model_kwargs"]),
    )
    null_hypos = _load_null_hypos(db) if g is None else g.null_hypos
    cal_algo = AdaCalibration(db, model, null_hypos, c)

    if g is not None:
        # Copy the input grid so that the caller is not surprised by any changes.
        df = copy.deepcopy(g.df)
        df["K"] = c["init_K"]
        df["step_id"] = 0
        df["step_iter"], n_packets = step_iter_assignments(df, c["packet_size"])
        df["creator_id"] = worker_id
        df["creation_time"] = simple_timer()

        db.init_tiles(df)
        _store_null_hypos(db, null_hypos)

        n_tiles = df.shape[0]
        logger.debug(
            f"first step {(0, 0, n_packets, n_tiles)} "
            f"n_tiles={n_tiles} packet_size={c['packet_size']}"
        )
        db.set_step_info(step_id=0, step_iter=0, n_iter=n_packets, n_tiles=n_tiles)

    ########################################
    # Run adagrid!
    ########################################
    if c["n_iter"] == 0:
        return 0, [], db

    reports = []
    for worker_iter in range(c["n_iter"]):
        try:
            report = dict(worker_iter=worker_iter, worker_id=worker_id)
            start = time.time()
            status, report = distributed.run(cal_algo, db, report, c["n_steps"])
            report["status"] = status.name
            report["runtime_full_iter"] = time.time() - start

            if callback is not None:
                callback(worker_iter, report, db)
            reports.append(report)

            if status.done():
                # We just stop this worker. The other workers will continue to run
                # but will stop once they reached the convergence criterion check.
                # It might be faster to stop all the workers immediately, but that
                # would be more engineering effort.
                break
        except KeyboardInterrupt:
            # TODO: we want to die robustly when there's a keyboard interrupt.
            print("KeyboardInterrupt")
            return worker_iter, reports, db

    return worker_iter + 1, reports, db


def verify_adagrid(df):
    inactive_ids = df.loc[~df["active"], "id"]
    assert inactive_ids.unique().shape == inactive_ids.shape

    parents = df["parent_id"].unique()
    parents_that_dont_exist = np.setdiff1d(parents, inactive_ids)
    inactive_tiles_with_no_children = np.setdiff1d(inactive_ids, parents)
    assert parents_that_dont_exist.shape[0] == 1
    assert parents_that_dont_exist[0] == 0
    assert inactive_tiles_with_no_children.shape[0] == 0
