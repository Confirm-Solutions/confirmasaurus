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
    seed=c.model_seed,
    max_K=c.init_K * 2**c.n_K_double,
    **json.loads(c.model_kwargs),
)
driver = AdaCalibrationDriver(None, model, null_hypos, c)
driver.bootstrap_calibrate(gtemp.df, 0.025)
gtemp.df['K'].value_counts()
```

"""
import copy
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pprint import pformat

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import imprint.log
from .db import DuckDBTiles
from confirm.adagrid.convergence import WorkerStatus
from imprint import batching
from imprint import driver
from imprint import grid
from imprint.timer import simple_timer

logger = imprint.log.getLogger(__name__)


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
        self.packet_size = self.c.packet_size

    def bootstrap_calibrate(self, df, alpha, tile_batch_size=None):
        tile_batch_size = (
            self.tile_batch_size if tile_batch_size is None else tile_batch_size
        )

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
            K_g = grid.Grid(K_df, self.c.worker_id)

            theta, vertices = K_g.get_theta_and_vertices()
            bootstrap_lams, alpha0 = batching.batch(
                _batched,
                tile_batch_size,
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
            K_g = grid.Grid(K_df, self.c.worker_id)

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

    def process_tiles(self, *, tiles_df):
        # This method actually runs the calibration and bootstrapping.
        # It is called once per iteration.
        # Several auxiliary fields are calculated because they are needed for
        # selecting the next iteration's tiles: impossible and orderer

        lams_df = self.bootstrap_calibrate(tiles_df, self.alpha)
        lams_df.insert(0, "processor_id", self.c.worker_id)
        lams_df.insert(1, "processing_time", simple_timer())
        lams_df.insert(2, "eligible", True)

        # we use insert here to order columns nicely for reading raw data
        lams_df.insert(3, "grid_cost", self.alpha - lams_df["alpha0"])
        lams_df.insert(
            4,
            "impossible",
            lams_df["alpha0"] < (self.calibration_min_idx + 1) / (tiles_df["K"] + 1),
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

    def step(self, worker_iter):
        """
        One iteration of the adagrid algorithm.

        Parallelization:
        There are four main portions of the adagrid algo:
        - Convergence criterion --> serialize to avoid data races.
        - Tile selection --> serialize to avoid data races.
        - Deciding whether to refine or deepen --> parallelize.
        - Simulation and CSE --> parallelize.
        By using a distributed lock around the convergence/tile selection, we can have
        equality between workers and not need to have a leader node.

        Args:
            i: The iteration number.

        Returns:
            done: A value from the Enum Convergence:
                 (INCOMPLETE, CONVERGED, FAILED).
            report: A dictionary of information about the iteration.
        """

        start = time.time()
        self.report = dict(worker_iter=worker_iter, worker_id=self.c.worker_id)

        ########################################
        # Step 1: Get the next batch of tiles to process. We do this before
        # checking for convergence because part of the convergence criterion is
        # whether there are any impossible tiles.
        ########################################
        max_loops = 25
        i = 0
        status = WorkerStatus.WORK
        self.report["runtime_wait_for_lock"] = 0
        self.report["runtime_wait_for_work"] = 0
        while i < max_loops:
            self.report["waitings"] = i
            with self.db.lock:
                self.report["runtime_wait_for_lock"] += time.time() - start
                start = time.time()

                step_id, step_iter, step_n_iter, step_n_tiles = self.db.get_step_info()
                self.report["step_id"] = step_id
                self.report["step_iter"] = step_iter
                self.report["step_n_iter"] = step_n_iter
                self.report["step_n_tiles"] = step_n_tiles

                # Check if there are iterations left in this step.
                # If there are, get the next batch of tiles to process.
                if step_iter < step_n_iter:
                    logger.debug(f"get_work(step_id={step_id}, step_iter={step_iter})")
                    work = self.db.get_work(step_id, step_iter)
                    self.report["runtime_get_work"] = time.time() - start
                    self.report["work_extraction_time"] = time.time()
                    self.report["n_processed"] = work.shape[0]
                    logger.debug(f"get_work(...) returned {work.shape[0]} tiles")

                    # If there's work, return it!
                    if work.shape[0] > 0:
                        self.db.set_step_info(
                            step_id=step_id,
                            step_iter=step_iter + 1,
                            n_iter=step_n_iter,
                            n_tiles=step_n_tiles,
                        )
                        logger.debug("Returning %s tiles.", work.shape[0])
                        return status, work, self.report
                    else:
                        # If step_iter < step_n_iter but there's no work, then
                        # The INSERT into tiles that was supposed to populate
                        # the work is probably incomplete. We should wait a
                        # very short time and try again.
                        logger.debug("No work despite step_iter < step_n_iter.")
                        wait = 0.1

                # If there are no iterations left in the step, we check if the
                # step is complete. For a step to be complete means that all
                # tiles have results.
                else:
                    n_processed_tiles = self.db.n_processed_tiles(step_id)
                    self.report["n_finished_tiles"] = n_processed_tiles
                    if n_processed_tiles == step_n_tiles:
                        # If a packet has just been completed, we check for convergence.
                        status = self.convergence_criterion()
                        self.report["runtime_convergence_criterion"] = (
                            time.time() - start
                        )
                        start = time.time()
                        if status:
                            logger.debug("Convergence!!")
                            return WorkerStatus.CONVERGED, None, self.report

                        if step_id >= self.c.n_steps - 1:
                            # We've completed all the steps, so we're done.
                            logger.debug("Reached max number of steps. Terminating.")
                            return WorkerStatus.REACHED_N_STEPS, None, self.report

                        # If we haven't converged, we create a new step.
                        new_step_id = self.new_step(step_id + 1)

                        self.report["runtime_new_step"] = time.time() - start
                        start = time.time()
                        if new_step_id == "empty":
                            # New packet is empty so we have terminated but
                            # failed to converge.
                            logger.debug(
                                "New packet is empty. Terminating despite "
                                "failure to converge."
                            )
                            return WorkerStatus.FAILED, None, self.report
                        else:
                            # Successful new packet. We should check for work again
                            # immediately.
                            status = WorkerStatus.NEW_STEP
                            wait = 0
                            logger.debug("Successfully created new packet.")
                    else:
                        # No work available, but the packet is incomplete. This is
                        # because other workers have claimed all the work but have not
                        # processsed yet.
                        # In this situation, we should release the lock and wait for
                        # other workers to finish.
                        wait = 1
                        logger.debug("No work available, but packet is incomplete.")
            if wait > 0:
                logger.debug("Waiting %s seconds and checking for work again.", wait)
                time.sleep(wait)
            if i > 2:
                logger.warning(
                    f"Worker {self.c.worker_id} has been waiting for work for"
                    f" {i} iterations. This might indicate a bug."
                )
            self.report["runtime_wait_for_work"] += time.time() - start
            i += 1

        return WorkerStatus.STUCK, None, self.report

    def convergence_criterion(self):
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
        worst_tile_tie_sum = self.many_rej(
            worst_tile, np.array([lamss] + list(B_lamss))
        ).iloc[0]
        worst_tile_tie_est = worst_tile_tie_sum / worst_tile["K"].iloc[0]

        # Given these TIE values, we can compute bias, standard deviation and
        # spread.
        bias_tie = worst_tile_tie_est[0] - worst_tile_tie_est[1:].mean()
        std_tie = worst_tile_tie_est.std()
        spread_tie = worst_tile_tie_est.max() - worst_tile_tie_est.min()
        grid_cost = worst_tile["grid_cost"].iloc[0]

        self.report.update(
            dict(
                bias_tie=bias_tie,
                std_tie=std_tie,
                spread_tie=spread_tie,
                grid_cost=grid_cost,
                lamss=lamss,
            )
        )
        self.report["min(B_lamss)"] = min(B_lamss)
        self.report["max(B_lamss)"] = max(B_lamss)
        self.report["tie_{k}(lamss)"] = worst_tile_tie_est[0]
        self.report["tie + slack"] = worst_tile_tie_est[0] + grid_cost + bias_tie

        # The convergence criterion itself.
        self.report["converged"] = (
            (bias_tie < self.bias_target)
            and (std_tie < self.std_target)
            and (grid_cost < self.grid_target)
        )
        return self.report["converged"]

    def new_step(self, new_step_id):
        tiles = self.db.select_tiles(self.c.step_size, "orderer")
        logger.info(
            f"Preparing new step {new_step_id} with {tiles.shape[0]} parent tiles."
        )
        tiles["finisher_id"] = self.c.worker_id
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
        twb_worst_tile_lams = self.bootstrap_calibrate(
            twb_worst_tile, self.alpha, tile_batch_size=1
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
        tiles["refine"] = (tiles["grid_cost"] > self.grid_target) & (
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
        self.report.update(
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

        df = refine_deepen(tiles, self.null_hypos, self.max_K, self.c.worker_id).df
        df["step_id"] = new_step_id
        df["step_iter"], n_packets = step_iter_assignments(df, self.c.packet_size)
        df["creator_id"] = self.c.worker_id
        df["creation_time"] = simple_timer()

        n_tiles = df.shape[0]
        logger.debug(
            f"new step {(new_step_id, 0, n_packets, n_tiles)} "
            f"n_tiles={n_tiles} packet_size={self.c.packet_size}"
        )
        self.db.set_step_info(
            step_id=new_step_id, step_iter=0, n_iter=n_packets, n_tiles=n_tiles
        )

        self.db.insert_tiles(df)
        self.report.update(
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


def print_report(_iter, report, _ada):
    ready = report.copy()
    for k in ready:
        if (
            isinstance(ready[k], float)
            or isinstance(ready[k], np.floating)
            or isinstance(ready[k], jnp.DeviceArray)
        ):
            ready[k] = f"{ready[k]:.6f}"
    logger.debug(pformat(ready))


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
    return _run(["pip", "freeze"])


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
    n_steps=100,
    step_size=2**10,
    n_iter=100,
    packet_size=None,
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
    n_steps: int
    step_size: int
    n_iter: int
    packet_size: int
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

        if self.packet_size is None:
            self.packet_size = self.step_size

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
    n_steps: int = None,
    step_size: int = None,
    n_iter: int = None,
    packet_size: int = None,
    std_target: float = None,
    calibration_min_idx: int = None,
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
        calibration_min_idx: The minimum calibration selection index. We enforce that:
                        `alpha0 >= (calibration_min_idx + 1) / (K + 1)`
                        A larger value will reduce the variance of lambda* but
                        will require more computational effort because K and/or
                        alpha0 will need to be larger. Defaults to 40.
        prod: Is this a production run? If so, we will collection extra system
              configuration info. Setting this to False will make startup time
              a bit faster. Defaults to True.
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
    imprint.log.worker_id.set(worker_id)

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
        n_steps,
        step_size,
        n_iter,
        packet_size,
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
        df = copy.deepcopy(g.df)
        df["K"] = c.init_K
        df["step_id"] = 0
        df["step_iter"], n_packets = step_iter_assignments(df, c.packet_size)
        df["creator_id"] = worker_id
        df["creation_time"] = simple_timer()

        db.init_tiles(df)
        _store_null_hypos(db, null_hypos)

        n_tiles = df.shape[0]
        logger.debug(
            f"first step {(0, 0, n_packets, n_tiles)} "
            f"n_tiles={n_tiles} packet_size={c.packet_size}"
        )
        db.set_step_info(step_id=0, step_iter=0, n_iter=n_packets, n_tiles=n_tiles)

    ########################################
    # Run adagrid!
    ########################################
    if c.n_iter == 0:
        return 0, [], db

    reports = []
    for worker_iter in range(c.n_iter):
        try:
            start = time.time()
            status, work, report = ada_driver.step(worker_iter)
            report["runtime_not_processing"] = time.time() - start
            report["status"] = status.name
            if work is not None and work.shape[0] > 0:
                start = time.time()
                results = ada_driver.process_tiles(tiles_df=work)
                report["runtime_processing"] = time.time() - start
                db.insert_results(results)

            if callback is not None:
                callback(worker_iter, report, ada_driver)
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
