import json
import platform
import subprocess
from dataclasses import dataclass

import jax
import pandas as pd


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
