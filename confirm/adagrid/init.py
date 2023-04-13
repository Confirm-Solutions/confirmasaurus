import copy
import json
import logging
import platform
import subprocess
import time
import warnings

import jax
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def init(db, algo_type, kwargs):
    g = kwargs["g"]

    cfg = first(kwargs)

    add_system_cfg(cfg)

    tiles_df = init_grid(g, db, cfg)

    cfg_copy = copy.copy(cfg)
    for k in ["git_diff", "conda_list", "nvidia_smi", "pip_freeze"]:
        if k in cfg_copy:
            del cfg_copy[k]
    logger.info("Config minus system info: \n%s", cfg_copy)

    cfg["model_kwargs"] = json.loads(cfg["model_kwargs_json"])
    algo = algo_type(kwargs["model_type"], db, cfg, kwargs["callback"])

    db.expected_counts[0] = dict(tiles=0, results=0, done=0, groups=0)

    return algo, tiles_df


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
    return cfg


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
    return cfg


def add_system_cfg(cfg):
    cfg["init_time"] = time.time()
    cfg["jax_platform"] = jax.lib.xla_bridge.get_backend().platform

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
    if cfg["record_system"]:
        cfg["pip_freeze"] = _run(["pip", "freeze"])
        cfg["conda_list"] = _run(["conda", "list"])
    else:
        cfg["pip_freeze"] = "skipped because record_system=False"
        cfg["conda_list"] = "skipped because record_system=False"

    cfg["max_K"] = cfg["init_K"] * 2 ** cfg["n_K_double"]


def init_grid(g, db, cfg):
    # Copy the input grid so that the caller is not surprised by any changes.
    df = copy.deepcopy(g.df)
    df["K"] = cfg["init_K"]

    df["step_id"] = 0
    df["packet_id"] = assign_packets(df, cfg["packet_size"])
    df.rename(columns={"active": "active_at_birth"}, inplace=True)

    # ID 0 is reserved for parents of the initial tiles.
    df["parent_id"] = 0

    db.insert("config", pd.DataFrame([cfg]), create=True)
    done_df = pd.DataFrame(
        {
            "step_id": pd.Series(dtype="int"),
            "group_id": pd.Series(dtype="int"),
            "count": pd.Series(dtype="int"),
        }
    )
    db.insert("done", done_df)

    n_packets = df["packet_id"].nunique()
    logger.debug(
        "Initialized database with %d tiles and %d null hypos."
        " The tiles are split into %d packets with packet_size=%s.",
        df.shape[0],
        len(g.null_hypos),
        n_packets,
        cfg["packet_size"],
    )

    return df


def assign_packets(df, packet_size):
    cum_sims = df["K"].cumsum()
    return pd.Series((cum_sims // packet_size).astype(int), df.index)


def _run(cmd):
    try:
        return (
            subprocess.check_output(" ".join(cmd), stderr=subprocess.STDOUT, shell=True)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.returncode} {exc.output}"
