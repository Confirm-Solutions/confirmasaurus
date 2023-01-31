import json
import logging
import platform
import subprocess
from pprint import pformat

import jax
import jax.numpy as jnp
import numpy as np

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


def prepare_config(cfg_dict, overrides, worker_id, prod):
    cfg = dict()
    cfg["worker_id"] = worker_id
    cfg["prod"] = prod

    for k in cfg_dict:
        if k in overrides:
            cfg[k] = overrides[k]
        else:
            cfg[k] = cfg_dict[k]

    for k, v in get_system_info(prod).items():
        cfg[k] = v

    cfg["tile_batch_size"] = cfg["tile_batch_size"] or (
        64 if cfg["jax_backend"] == "gpu" else 4
    )

    if cfg["model_kwargs"] is None:
        cfg["model_kwargs"] = {}
        # TODO: is json suitable for all models? are there models that are going to
        # want to have large non-jsonable objects as parameters?
    cfg["model_kwargs"] = json.dumps(cfg["model_kwargs"])

    if cfg["packet_size"] is None:
        cfg["packet_size"] = cfg["step_size"]

    return cfg


def get_system_info(prod: bool):
    out = dict(
        git_hash=_get_git_revision_hash(),
        git_diff=_get_git_diff(),
        platform=platform.platform(),
        nvidia_smi=_get_nvidia_smi(),
        jax_backend=jax.lib.xla_bridge.get_backend().platform,
    )
    if prod:
        out["pip_freeze"] = _get_pip_freeze()
        out["conda_list"] = _get_conda_list()
    else:
        out["pip_freeze"] = "skipped for non-prod run"
        out["conda_list"] = "skipped for non-prod run"
    return out


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
