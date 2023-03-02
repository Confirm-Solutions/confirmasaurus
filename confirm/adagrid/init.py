import asyncio
import codecs
import copy
import json
import logging
import platform
import subprocess
import warnings

import cloudpickle
import jax
import numpy as np
import pandas as pd

import imprint as ip
from confirm.adagrid.db import DuckDBTiles

logger = logging.getLogger(__name__)


async def init(algo_type, is_leader, worker_id, n_zones, kwargs):
    db = kwargs.get("db", None)
    g = kwargs.get("g", None)
    ip.log.worker_id.set(worker_id)

    assert (db is not None) or is_leader

    if db is None and g is None:
        raise ValueError("If no grid is provided, a database must be provided.")

    if db is None:
        db = DuckDBTiles.connect()
        tiles_exists = False
    else:
        tiles_exists = db.does_table_exist("tiles")

    if g is None and not tiles_exists:
        raise ValueError("If no grid is provided, the database must contain tiles.")

    if g is not None and not tiles_exists:
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

    if g is not None and not tiles_exists and is_leader:
        wait_for_grid, incomplete_packets, zone_steps = await init_grid(
            g, db, cfg, n_zones
        )
        wait_for.extend(wait_for_grid)
    elif is_leader:
        if g is not None:
            logger.warning(
                "Ignoring grid because tiles already exist in the provided database."
            )
        incomplete_packets = db.get_incomplete_packets()
        zone_steps = db.get_zone_steps()
    else:
        incomplete_packets = None
        zone_steps = None

    model_kwargs = json.loads(cfg["model_kwargs_json"])
    model = kwargs["model_type"](
        seed=cfg["model_seed"],
        max_K=cfg["init_K"] * 2 ** cfg["n_K_double"],
        **model_kwargs,
    )
    algo = algo_type(model, null_hypos, db, cfg, kwargs["callback"])

    await asyncio.gather(*wait_for)

    return algo, incomplete_packets, zone_steps


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
    if overrides is None:
        overrides = {}

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

    df["coordination_id"] = 0
    df["step_id"] = 0
    df["zone_id"] = assign_tiles(g.n_tiles, n_zones)
    df["packet_id"] = assign_packets(df, cfg["packet_size"])
    df["creator_id"] = 1
    df["creation_time"] = ip.timer.simple_timer()

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

    incomplete_packets = []
    for zone_id, zone in df.groupby("zone_id"):
        incomplete_packets.extend(
            [(zone_id, 0, p) for p in range(zone["packet_id"].max() + 1)]
        )
    zone_steps = {i: 0 for i, zone in df.groupby("zone_id")}
    return wait_for, incomplete_packets, zone_steps


def assign_tiles(n_tiles, n_zones):
    splits = np.array_split(np.arange(n_tiles), n_zones)
    assignment = np.empty(n_tiles, dtype=np.uint32)
    for i in range(n_zones):
        assignment[splits[i]] = i
    return assignment


def assign_packets(df, packet_size):
    def f(df):
        return pd.Series(
            np.floor(np.arange(df.shape[0]) / packet_size).astype(int),
            df.index,
        )

    return df.groupby("zone_id")["zone_id"].transform(f)


def _store_null_hypos(db, null_hypos):
    # we need to convert the pickled object to a valid string so that it can be
    # inserted into a database. converting to a from base64 achieves this goal:
    # https://stackoverflow.com/a/30469744/3817027
    serialized = [
        codecs.encode(cloudpickle.dumps(h), "base64").decode() for h in null_hypos
    ]
    desc = [h.description() for h in null_hypos]
    df = pd.DataFrame({"serialized": serialized, "description": desc})
    db.store.set("null_hypos", df)


def _load_null_hypos(db):
    df = db.store.get("null_hypos")
    null_hypos = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        null_hypos.append(
            cloudpickle.loads(codecs.decode(row["serialized"].encode(), "base64"))
        )
    return null_hypos


def _run(cmd):
    try:
        return (
            subprocess.check_output(" ".join(cmd), stderr=subprocess.STDOUT, shell=True)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.returncode} {exc.output}"


async def _launch_task(db, f, *args, in_thread=True, **kwargs):
    if in_thread and db.supports_threads:
        coro = asyncio.to_thread(f, *args, **kwargs)
    elif in_thread and not db.supports_threads:
        out = f(*args, **kwargs)

        async def _coro():
            return out

        coro = _coro()
    else:
        coro = f(*args, **kwargs)
    task = asyncio.create_task(coro)
    # Sleep immediately to allow the task to start.
    await asyncio.sleep(0)
    return task
