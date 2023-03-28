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
from .const import MAX_STEP

logger = logging.getLogger(__name__)


def init(algo_type, kwargs):
    db = kwargs["db"]
    g = kwargs.get("g", None)

    tiles_exists = db.does_table_exist("tiles")
    if g is None and not tiles_exists:
        raise ValueError("If no grid is provided, the database must contain tiles.")
    if g is not None and tiles_exists:
        raise ValueError("If a grid is provided, the database must not contain tiles.")

    if g is not None and not tiles_exists:
        cfg, null_hypos = first(kwargs)
    else:
        cfg, null_hypos = join(db, kwargs)

    add_system_cfg(cfg)

    if g is not None and not tiles_exists:
        incomplete_packets = init_grid(g, db, cfg)
        next_step = 1
    else:
        db.insert_config(pd.DataFrame([cfg]))
        if g is not None:
            logger.warning(
                "Ignoring grid because tiles already exist " "in the provided database."
            )
        incomplete_packets = db.get_incomplete_packets()
        next_step = db.get_next_step()

    cfg_copy = copy.copy(cfg)
    del cfg_copy["git_diff"]
    del cfg_copy["conda_list"]
    del cfg_copy["nvidia_smi"]
    del cfg_copy["pip_freeze"]
    logger.info("Config (minus system info): \n%s", cfg_copy)

    cfg["model_kwargs"] = json.loads(cfg["model_kwargs_json"])
    model = kwargs["model_type"](
        seed=cfg["model_seed"],
        max_K=cfg["init_K"] * 2 ** cfg["n_K_double"],
        **cfg["model_kwargs"],
    )
    algo = algo_type(model, null_hypos, db, cfg, kwargs["callback"])

    return algo, incomplete_packets, next_step


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
    load_cfg_df = db.get_config()
    cfg = load_cfg_df.iloc[0].to_dict()

    for k in ["tile_batch_size", "job_name"]:
        if np.isnan(cfg[k]):
            cfg[k] = None

    # IMPORTANT: Except for overrides, entries in kwargs will be ignored!
    overrides = kwargs["overrides"]
    if overrides is None:
        overrides = {}

    for k in overrides:
        # Some parameters cannot be overridden because the job just wouldn't
        # make sense anymore.
        if k in [
            "lam",
            "model_seed",
            "model_kwargs",
            "delta",
            "init_K",
            "n_K_double",
            "alpha",
            "bootstrap_seed",
            "nB",
            "model_name",
            "n_parallel_steps",
        ]:
            raise ValueError(f"Parameter {k} cannot be overridden.")
        cfg[k] = overrides[k]
    null_hypos = _deserialize_null_hypos(db.get_null_hypos())
    return cfg, null_hypos


def add_system_cfg(cfg):
    cfg["jax_platform"] = jax.lib.xla_bridge.get_backend().platform
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


def init_grid(g, db, cfg):
    # Copy the input grid so that the caller is not surprised by any changes.
    df = copy.deepcopy(g.df)
    df["K"] = cfg["init_K"]

    df["step_id"] = 0
    df["packet_id"] = assign_packets(df, cfg["packet_size"])
    df["creation_time"] = ip.timer.simple_timer()
    df["inactivation_step"] = MAX_STEP
    df.drop("active", axis=1, inplace=True)

    null_hypos_df = _serialize_null_hypos(g.null_hypos)

    db.init_grid(df, null_hypos_df, pd.DataFrame([cfg]))

    n_packets = df["packet_id"].nunique()
    logger.debug(
        "Initialized database with %d tiles and %d null hypos."
        " The tiles are split into %d packets with packet_size=%s.",
        df.shape[0],
        len(g.null_hypos),
        n_packets,
        cfg["packet_size"],
    )

    incomplete_packets = [(0, i) for i in range(n_packets)]
    return incomplete_packets


def assign_packets(df, packet_size):
    return pd.Series(
        np.floor(np.arange(df.shape[0]) / packet_size).astype(int),
        df.index,
    )


def _serialize_null_hypos(null_hypos):
    # we need to convert the pickled object to a valid string so that it can be
    # inserted into a database. converting to a from base64 achieves this goal:
    # https://stackoverflow.com/a/30469744/3817027
    serialized = [
        codecs.encode(cloudpickle.dumps(h), "base64").decode() for h in null_hypos
    ]
    desc = [h.description() for h in null_hypos]
    return pd.DataFrame({"serialized": serialized, "description": desc})


def _deserialize_null_hypos(df):
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
