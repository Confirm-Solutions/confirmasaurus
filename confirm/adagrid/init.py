import asyncio
import copy
import json
import logging
import platform
import warnings

import jax
import pandas as pd

import imprint.timer
from confirm.adagrid.adagrid import _launch_task
from confirm.adagrid.adagrid import _load_null_hypos
from confirm.adagrid.adagrid import _run
from confirm.adagrid.adagrid import _store_null_hypos
from confirm.adagrid.adagrid import assign_packets
from confirm.adagrid.adagrid import assign_tiles
from confirm.adagrid.db import DuckDBTiles

logger = logging.getLogger(__name__)


async def init(algo_type, n_zones, kwargs):
    db = kwargs["db"]
    g = kwargs["g"]

    if db is None and g is None:
        raise ValueError("Must provide an initial grid or an existing database!")

    if db is None:
        db = DuckDBTiles.connect()

    worker_id = kwargs.get("worker_id", None)
    if worker_id is None:
        worker_id = db.new_workers(1)[0]
    imprint.log.worker_id.set(worker_id)

    if g is not None:
        cfg, null_hypos = first(kwargs)
    else:
        cfg, null_hypos = join(db, kwargs)

    cfg["worker_id"] = worker_id
    add_system_cfg(cfg)
    # we wrap set_or_append in a lambda so that the db.store access can be
    # run in a separate thread in case the Store __init__ method does any
    # substantial work (e.g. in ClickhouseStore, we create a table)
    wait_for = [
        await _launch_task(
            db, lambda df: db.store.set_or_append("config", df), pd.DataFrame([cfg])
        )
    ]

    if g is not None:
        wait_for_grid, zone_info = await init_grid(g, db, cfg, n_zones)
        wait_for.extend(wait_for_grid)
    else:
        # TODO: not sure about this situation...
        zone_info = None

    model_kwargs = json.loads(cfg["model_kwargs_json"])
    model = kwargs["model_type"](
        seed=cfg["model_seed"],
        max_K=cfg["init_K"] * 2 ** cfg["n_K_double"],
        **model_kwargs,
    )
    algo = algo_type(model, null_hypos, db, cfg, kwargs["callback"])

    await asyncio.gather(*wait_for)

    return algo, zone_info


def first(kwargs):
    # Using locals() as kwargs is a simple way to get all the config
    # vars in the function definition. But, we need to erase fields
    # that are not part of the "config".
    cfg = {
        k: v
        for k, v in kwargs.items()
        if k
        not in [
            "model_type",
            "g",
            "db",
            "overrides",
            "callback",
            "model_kwargs",
            "transformation",
            "backend",
        ]
    }

    # Storing the model_type is infeasible because it's a class, so we store
    # the model type name instead.
    cfg["model_name"] = kwargs["model_type"].__name__

    model_kwargs = kwargs["model_kwargs"]
    if model_kwargs is None:
        model_kwargs = {}
    cfg["model_kwargs_json"] = json.dumps(model_kwargs)

    if kwargs["overrides"] is not None:
        warnings.warn("Overrides are ignored when starting a new job.")
    return cfg, kwargs["g"].null_hypos


def join(db, kwargs):
    # If we are resuming a job, we need to load the config from the database.
    load_cfg_df = db.store.get("config")
    cfg = load_cfg_df.iloc[0].to_dict()

    # IMPORTANT: Except for overrides, entries in kwargs will be ignored!
    overrides = kwargs["overrides"]
    for k in overrides:
        # Some parameters cannot be overridden because the job just wouldn't
        # make sense anymore.
        if k in [
            "model_seed",
            "model_kwargs",
            "alpha",
            "init_K",
            "n_K_double",
            "bootstrap_seed",
            "nB",
            "model_name",
        ]:
            raise ValueError(f"Parameter {k} cannot be overridden.")
        cfg[k] = overrides[k]
    return cfg, _load_null_hypos(db)


def add_system_cfg(cfg):
    cfg["jax_platform"] = jax.lib.xla_bridge.get_backend().platform
    default_tile_batch_size = dict(gpu=64, cpu=4)
    cfg["tile_batch_size"] = cfg["tile_batch_size"] or (
        default_tile_batch_size[cfg["jax_platform"]]
    )

    if cfg["packet_size"] is None:
        cfg["packet_size"] = cfg["step_size"]

    ########################################
    # STEP 3: Collect a bunch of system information for later debugging and
    # reproducibility.
    ########################################
    cfg.update(
        dict(
            git_hash=_run(["git", "rev-parse", "HEAD"]),
            git_diff=_run(["git", "diff", "HEAD"]),
            platform=platform.platform(),
            nvidia_smi=_run(["nvidia-smi"]),
        )
    )
    if cfg["prod"]:
        cfg["pip_freeze"] = _run(["pip", "freeze"])
        cfg["conda_list"] = _run(["conda", "list"])
    else:
        cfg["pip_freeze"] = "skipped because prod=False"
        cfg["conda_list"] = "skipped because prod=False"

    cfg["max_K"] = cfg["init_K"] * 2 ** cfg["n_K_double"]


async def init_grid(g, db, cfg, n_zones):
    # Copy the input grid so that the caller is not surprised by any changes.
    df = copy.deepcopy(g.df)
    df["K"] = cfg["init_K"]

    df["step_id"] = 0
    df["zone_id"] = assign_tiles(g.n_tiles, n_zones, range(n_zones))
    df["packet_id"] = assign_packets(df, cfg["packet_size"])
    df["creator_id"] = 1
    df["creation_time"] = imprint.timer.simple_timer()

    wait_for = [
        await _launch_task(db, db.init_tiles, df, in_thread=False),
        await _launch_task(db, _store_null_hypos, db, g.null_hypos),
    ]

    logger.debug(
        "Initialized database with %d tiles and %d null hypos."
        " The tiles are split between %d zones with packet_size=%s.",
        df.shape[0],
        len(g.null_hypos),
        n_zones,
        cfg["packet_size"],
    )

    zone_info = dict()
    for zone_id, zone in df.groupby("zone_id"):
        zone_info[zone_id] = zone["packet_id"].max() + 1
    return wait_for, zone_info
