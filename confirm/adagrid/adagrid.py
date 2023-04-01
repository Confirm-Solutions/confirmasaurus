import abc
import contextlib
import logging
import queue
import time
from pprint import pformat

import jax
import numpy as np

import imprint as ip
from .asyncio_runner import Runner
from .const import MAX_STEP
from .db import DatabaseLogging
from .db import DuckDBTiles
from .init import init
from .step import new_step
from .step import process_initial_packets
from .step import submit_packet_df
from .step import wait_for_packets
from .step import WorkerStatus

logger = logging.getLogger(__name__)


def pass_control_to_backend(algo_type, kwargs):
    backend = kwargs.get("backend", None)
    if backend is None:
        backend = LocalBackend()
    kwargs.update(backend.get_cfg())

    return backend.entrypoint(algo_type, kwargs)


def setup_db(cfg, require_fresh=False):
    if cfg["job_name"] is None:
        if cfg["job_name_prefix"] is None:
            cfg["job_name_prefix"] = "unnamed"
        cfg["job_name"] = f"{cfg['job_name_prefix']}_{time.strftime('%Y%m%d_%H%M%S')}"

    print(list(cfg.keys()))
    ch_client = None
    if cfg["clickhouse_service"] is not None:
        import confirm.cloud.clickhouse as ch

        ch_client = ch.connect(
            job_name=cfg["job_name"], service=cfg["clickhouse_service"]
        )
        if require_fresh and ch.does_table_exist(ch_client, "tiles"):
            raise ValueError(
                "ClickHouse table 'tiles' already exists. "
                "Please choose a fresh job_name."
            )

    if cfg["job_name"].startswith("unnamed"):
        db_filepath = ":memory:"
    else:
        db_filepath = cfg["job_name"] + ".db"

    db = DuckDBTiles.connect(path=db_filepath, ch_client=ch_client)
    if require_fresh and db.does_table_exist("tiles"):
        raise ValueError(
            "DuckDB table 'tiles' already exists." " Please choose a fresh job_name."
        )
    return db


def entrypoint(backend, algo_type, kwargs):
    """
    Passing control of the entrypoint to the backend allows executing the
    leader somewhere besides the launching machine.

    The default behavior is to just run the leader locally.
    """

    db = setup_db(kwargs, require_fresh=True)
    with DatabaseLogging(db) as db_logging:
        with Runner() as runner:
            try:
                runner.run(async_entrypoint(backend, db, algo_type, kwargs))
            except Exception as e:  # noqa
                # bare except is okay because we want to log the exception. the
                # re-raise will preserve the original stack trace.
                logging.error("Adagrid error", exc_info=e)
                raise
            finally:
                # This should be the last thing we do so we can be sure that all logging
                # messages have been sent to the database.
                db_logging.flush()
                runner.run(db.ch_wait())
                assert len(db.ch_tasks) == 0
    return db


async def async_entrypoint(backend, db, algo_type, kwargs):
    entry_time = time.time()
    with timer("init"):
        algo, initial_tiles_df = init(db, algo_type, kwargs)
    next_step = 1

    start = time.time()
    async with backend.setup(algo):
        logger.debug(f"backend.setup() took {time.time() - start:.2f} seconds")

        with timer("process_initial_incompletes"):
            await process_initial_packets(backend, algo, initial_tiles_df)

        stopping_indicator = 0
        n_parallel_steps = algo.cfg["n_parallel_steps"]
        processing_tasks = queue.Queue()
        for step_id in range(next_step, algo.cfg["n_steps"]):
            basal_step_id = max(step_id - n_parallel_steps, 0)

            with timer("verify"):
                db.verify(basal_step_id)

            if time.time() - entry_time > kwargs["timeout"]:
                logger.info("Job timeout reached, stopping.")
                break

            with timer("new step"):
                logger.info(f"Beginning step {step_id}")
                status, tiles_df = new_step(algo, basal_step_id, step_id)

            if status in [WorkerStatus.CONVERGED, WorkerStatus.EMPTY_STEP]:
                stopping_indicator += 1
            else:
                stopping_indicator = 0

            if (
                status == WorkerStatus.CONVERGED
                and stopping_indicator >= n_parallel_steps
            ):
                logger.info("Converged. Stopping.")
                break
            elif (
                status == WorkerStatus.EMPTY_STEP
                and stopping_indicator >= n_parallel_steps
            ):
                logger.info(f"{n_parallel_steps} empty step. Stopping.")
                break

            with timer("process packets"):
                logger.info("Processing packets for step %d", step_id)
                processing_tasks.put(await submit_packet_df(backend, algo, tiles_df))
                if processing_tasks.qsize() > n_parallel_steps - 1:
                    await wait_for_packets(backend, algo, processing_tasks.get())

            with timer("Check on Clickhouse inserts"):
                await db.ch_checkup()

        with timer("process final packets"):
            while not processing_tasks.empty():
                await wait_for_packets(backend, algo, processing_tasks.get())

        with timer("verify"):
            db.verify(basal_step_id)

        assert processing_tasks.empty()
        assert step_id < algo.cfg["n_steps"]
        assert (
            step_id == algo.cfg["n_steps"] - 1 or stopping_indicator >= n_parallel_steps
        )


@contextlib.contextmanager
def timer(name):
    start = time.time()
    logger.debug(f"Starting timing block: {name}")
    yield
    logger.debug(f"{name} took {time.time() - start:.3f} seconds")


class Backend(abc.ABC):
    algo_cfg_entries = [
        "init_K",
        "n_K_double",
        "tile_batch_size",
        "lam",
        "delta",
        "global_target",
        "max_target",
        "bootstrap_seed",
        "nB",
        "alpha",
        "calibration_min_idx",
        "clickhouse_service",
        "job_name",
    ]

    def entrypoint(self, algo_type, kwargs):
        """
        Passing control of the entrypoint to the backend allows executing the
        leader somewhere besides the launching machine.

        The default behavior is to just run the leader locally.
        """
        return entrypoint(self, algo_type, kwargs)

    @abc.abstractmethod
    def get_cfg(self):
        pass

    @abc.abstractmethod
    @contextlib.asynccontextmanager
    async def setup(self, algo):
        pass

    @abc.abstractmethod
    def submit_tiles(self, tiles_df):
        pass

    @abc.abstractmethod
    async def wait_for_results(self, awaitable):
        pass


class LocalBackend(Backend):
    def get_cfg(self):
        return {}

    @contextlib.asynccontextmanager
    async def setup(self, algo):
        self.algo = algo
        yield

    def refine_deepen_sim_tiles(self, df, refine_deepen: bool):
        report = dict()
        if refine_deepen:
            start = time.time()

            step_id = df["step_id"].iloc[0]

            g_new = refine_and_deepen(df, self.algo.null_hypos, self.algo.cfg["max_K"])

            inactive_df = g_new.df[~g_new.df["active"]].copy()
            inactive_df["inactivation_step"] = step_id
            inactive_df["refine"] = 0
            inactive_df["deepen"] = 0
            inactive_df["split"] = True
            self.algo.db.insert_tiles(inactive_df)
            self.algo.db.insert_done(inactive_df)

            g_active = g_new.prune_inactive()
            tiles_df = g_active.df
            tiles_df.drop("active", axis=1, inplace=True)
            tiles_df["inactivation_step"] = MAX_STEP
            report["runtime_refine_deepen"] = time.time() - start
            report["n_inactive_tiles"] = inactive_df.shape[0]
        else:
            tiles_df = df

        self.algo.db.insert_tiles(tiles_df)

        report["n_tiles"] = tiles_df.shape[0]
        report["sim_start_time"] = time.time()
        tbs = self.algo.cfg["tile_batch_size"]
        if tbs is None:
            tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
        report["tile_batch_size"] = tbs
        start = time.time()
        results_df = self.algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)
        results_df["completion_step"] = MAX_STEP
        report["runtime_simulating"] = time.time() - start
        report["sim_done_time"] = time.time()

        self.algo.db.insert_results(results_df, self.algo.get_orderer())
        return report

    async def submit_tiles(self, tiles_df, refine_deepen: bool):
        return self.refine_deepen_sim_tiles(tiles_df, refine_deepen)

    async def wait_for_results(self, awaitable):
        return awaitable


def print_report(report, _db):
    ready = report.copy()
    for k in ready:
        if (
            isinstance(ready[k], float)
            or isinstance(ready[k], np.floating)
            or isinstance(ready[k], jax.Array)
        ):
            ready[k] = f"{ready[k]:.6f}"
    logger.debug(pformat(ready))


def refine_and_deepen(df, null_hypos, max_K):
    g_deepen_in = ip.Grid(df.loc[(df["deepen"] > 0) & (df["K"] < max_K)])
    g_deepen = ip.grid._raw_init_grid(
        g_deepen_in.get_theta(),
        g_deepen_in.get_radii(),
        parents=g_deepen_in.df["id"],
    )

    # We just multiply K by 2 to deepen.
    # TODO: it's possible to do better by multiplying by 4 or 8
    # sometimes when a tile clearly needs *way* more sims. how to
    # determine this?
    g_deepen.df["K"] = g_deepen_in.df["K"] * 2

    g_refine_in = ip.grid.Grid(df.loc[df["refine"] > 0])
    inherit_cols = ["K"]
    # TODO: it's possible to do better by refining by more than just a
    # factor of 2.
    g_refine = g_refine_in.refine(inherit_cols)

    # NOTE: Instead of prune_alternative here, we mark alternative tiles as
    # inactive. This means that we will have a full history of grid
    # construction.
    out = g_refine.concat(g_deepen)
    out = out.add_null_hypos(null_hypos, inherit_cols)
    out.df.loc[out._which_alternative(), "active"] = False
    for col in ["step_id", "packet_id"]:
        out.df[col] = df[col].iloc[0]
    return out
