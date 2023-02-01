import json
import logging
import platform
import subprocess
import warnings
from pprint import pformat

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _run(cmd):
    try:
        return (
            subprocess.check_output(" ".join(cmd), stderr=subprocess.STDOUT, shell=True)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.returncode} {exc.output}"


def prepare_config(db, locals_, worker_id):
    g = locals_["g"]
    overrides = locals_["overrides"]

    if g is None:
        load_cfg_df = db.store.get("config")
        cfg = load_cfg_df.iloc[0].to_dict()
        # Very important not to share worker_id between workers!!
        cfg["worker_id"] = None
        model_kwargs = json.loads(cfg["model_kwargs_json"])
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

    else:
        # Using locals() is a simple way to get all the config vars in the
        # function definition. But, we need to erase fields that are not part
        # of the "config".
        cfg = {
            k: v
            for k, v in locals_.items()
            if k
            not in ["modeltype", "g", "db", "overrides", "callback", "model_kwargs"]
        }
        cfg["model_name"] = locals_["modeltype"].__name__

        model_kwargs = locals_["model_kwargs"]
        if model_kwargs is None:
            model_kwargs = {}
        cfg["model_kwargs_json"] = json.dumps(model_kwargs)

        if overrides is not None:
            warnings.warn("Overrides are ignored when starting a new job.")

    cfg["worker_id"] = worker_id
    cfg["jax_backend"] = jax.lib.xla_bridge.get_backend().platform
    cfg["tile_batch_size"] = cfg["tile_batch_size"] or (
        64 if cfg["jax_backend"] == "gpu" else 4
    )

    if cfg["packet_size"] is None:
        cfg["packet_size"] = cfg["step_size"]

    cfg.update(
        dict(
            git_hash=_run(["git", "rev-parse", "HEAD"]),
            git_diff=_run(["git", "diff", "HEAD"]),
            platform=platform.platform(),
            nvidia_smi=_run(["nvidia-smi"]),
        )
    )
    if locals_["prod"]:
        cfg["pip_freeze"] = _run(["pip", "freeze"])
        cfg["conda_list"] = _run(["conda", "list"])
    else:
        cfg["pip_freeze"] = "skipped because prod=False"
        cfg["conda_list"] = "skipped because prod=False"

    cfg_df = pd.DataFrame([cfg])
    db.store.set_or_append("config", cfg_df)
    return cfg, model_kwargs


def print_report(_iter, report, _db):
    ready = report.copy()
    for k in ready:
        if (
            isinstance(ready[k], float)
            or isinstance(ready[k], np.floating)
            or isinstance(ready[k], jnp.DeviceArray)
        ):
            ready[k] = f"{ready[k]:.6f}"
    logger.debug(pformat(ready))
