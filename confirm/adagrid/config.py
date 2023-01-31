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


def prepare_config(cfg_dict, overrides, worker_id, prod):
    cfg = dict()
    cfg["worker_id"] = worker_id
    cfg["prod"] = prod

    for k in cfg_dict:
        if k in overrides:
            cfg[k] = overrides[k]
        else:
            cfg[k] = cfg_dict[k]

    cfg.update(
        dict(
            git_hash=_run(["git", "rev-parse", "HEAD"]),
            git_diff=_run(["git", "diff", "HEAD"]),
            platform=platform.platform(),
            nvidia_smi=_run(["nvidia-smi"]),
            jax_backend=jax.lib.xla_bridge.get_backend().platform,
        )
    )
    if prod:
        cfg["pip_freeze"] = _run(["pip", "freeze"])
        cfg["conda_list"] = _run(["conda", "list"])
    else:
        cfg["pip_freeze"] = "skipped because prod=False"
        cfg["conda_list"] = "skipped because prod=False"

    cfg["tile_batch_size"] = cfg["tile_batch_size"] or (
        64 if cfg["jax_backend"] == "gpu" else 4
    )

    if cfg["packet_size"] is None:
        cfg["packet_size"] = cfg["step_size"]

    return cfg


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
