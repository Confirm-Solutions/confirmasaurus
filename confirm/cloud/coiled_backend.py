import asyncio
import contextlib
import logging
import os
import subprocess
import tempfile

import dask
import jax
from distributed import get_worker
from distributed.diagnostics.plugin import UploadDirectory

import confirm
import imprint as ip
from ..adagrid.backend import Backend
from ..adagrid.backend import LocalBackend

logger = logging.getLogger(__name__)


def create_software_env():
    reqs = subprocess.run(
        "poetry export --without-hashes", stdout=subprocess.PIPE, shell=True
    ).stdout.decode("utf-8")
    reqs = reqs.split("\n")
    reqs = [r.split(";")[0][:-1] for r in reqs if "jax" not in r]
    confirm_dir = os.path.dirname(os.path.dirname(confirm.__file__))
    with open(os.path.join(confirm_dir, "requirements-coiled.txt"), "r") as f:
        reqs.extend([L.strip() for L in f.readlines()])
    reqs = [r for r in reqs if len(r) > 0]
    pip_installs = "\n".join([f"    - {r}" for r in reqs])
    environment_yml = f"""
name: confirm
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.10
  - nvidia:cuda-toolkit=11.8
  - conda-forge:cudnn=8.4
  - pip
  - pip:
{pip_installs}
"""
    logger.debug(f"Coiled environment:\n {environment_yml}")
    name = "confirm-coiled"
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(environment_yml)
        f.flush()
        logger.debug("Creating software environment %s", name)
        import coiled

        coiled.create_software_environment(name=name, conda=f.name)
    return name


def upload_pkg(client, module, restart=False):
    from pathlib import Path

    dir = Path(module.__file__).parent.parent
    skip = [os.path.join(dir, p) for p in os.listdir(dir) if p != module.__name__]

    def skip_f(fn):
        for s in skip:
            if fn.startswith(s):
                return True
        ext = os.path.splitext(fn)[1]
        return (ext == ".pyc") or (".DS_Store" in fn)

    return client.register_worker_plugin(
        UploadDirectory(dir, skip=(skip_f,), restart=restart, update_path=True),
        nanny=True,
    )


def setup_cluster(n_workers=1, idle_timeout="20 minutes"):
    create_software_env()
    import coiled

    cluster = coiled.Cluster(
        name="confirm-coiled",
        software="confirm-coiled",
        n_workers=n_workers,
        worker_vm_types=["g4dn.xlarge"],
        worker_gpu=1,
        compute_purchase_option="spot_with_fallback",
        shutdown_on_close=False,
        scheduler_options={"idle_timeout": idle_timeout},
    )
    cluster.scale(n_workers)
    client = cluster.get_client()
    reset_confirm_imprint(client)
    return cluster


def reset_confirm_imprint(client):
    upload_pkg(client, confirm)
    upload_pkg(client, ip, restart=True)


@dask.delayed
def check():
    # Report on the NVIDIA driver, CUDA version, and GPU model.
    import subprocess

    result = subprocess.run(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    nvidia_smi = result.stdout.decode("ascii")

    # Sometimes JAX is importable but not usable, so we check that it imports
    # and is usable.
    import jax

    jax_platform = jax.lib.xla_bridge.get_backend().platform
    key = jax.random.PRNGKey(0)
    _ = jax.random.uniform(key, (10000, 2)).mean(axis=0)

    # Check that confirm and imprint are importable.
    import confirm  # noqa
    import imprint  # noqa

    return nvidia_smi, jax_platform


def setup_worker(worker_args):
    worker = get_worker()

    (model_type, model_args, model_kwargs, algo_type, cfg) = worker_args
    # Hashing the arguments lets us avoid re-initializing the model and algo
    # if the arguments are the same.
    # NOTE: this could be extended to use an `algo` dictionary, where the key
    # is this hash. But, I'm suspicious that would cause out-of-memory errors
    # so the design is currently limited to a single algo per worker at a
    # single point in time.
    hash_args = hash(
        (
            model_type.__name__,
            model_args,
            tuple(model_kwargs.items()),
            algo_type.__name__,
            tuple(cfg.items()),
        )
    )
    has_hash = hasattr(worker, "algo_hash")
    has_algo = hasattr(worker, "algo")
    if not (has_algo and has_hash) or (has_hash and hash_args != worker.algo_hash):
        ip.package_settings()
        model = model_type(*model_args, **model_kwargs)
        cfg["worker_id"] = 2
        worker.algo = algo_type(model, None, None, cfg, None)
        worker.algo_hash = hash_args

        async def async_process_tiles(tiles_df):
            lb = LocalBackend()
            async with lb.setup(worker.algo):
                return await lb.process_tiles(tiles_df)

        import synchronicity

        synchronizer = synchronicity.Synchronizer()
        worker.process_tiles = synchronizer.create(async_process_tiles)[
            synchronicity.Interface.BLOCKING
        ]
    return worker.process_tiles


def dask_process_tiles(worker_args, packet_df):
    process_tiles = setup_worker(worker_args)
    jax_platform = jax.lib.xla_bridge.get_backend().platform
    assert jax_platform == "gpu"
    out, runtime_simulating = process_tiles(packet_df)
    return out, runtime_simulating


class CoiledBackend(Backend):
    def __init__(self, detach: bool = False, n_workers: int = 1, cluster=None):
        self.detach = detach
        self.n_workers = n_workers
        self.cluster = cluster

    def check(self):
        return check().compute()

    def get_cfg(self):
        return {}

    @contextlib.asynccontextmanager
    async def setup(self, algo):
        if self.cluster is None:
            self.cluster = setup_cluster(self.n_workers)
        self.client = self.cluster.get_client()
        algo_entries = [
            "init_K",
            "n_K_double",
            "tile_batch_size",
            "lam",
            "delta",
            "worker_id",
            "global_target",
            "max_target",
            "bootstrap_seed",
            "nB",
            "alpha",
            "calibration_min_idx",
        ]
        filtered_cfg = {k: v for k, v in algo.cfg.items() if k in algo_entries}
        worker_args = (
            type(algo.driver.model),
            (algo.cfg["model_seed"], algo.max_K),
            algo.cfg["model_kwargs"],
            type(algo),
            filtered_cfg,
        )
        self.worker_args_future = self.client.scatter(worker_args, broadcast=True)
        yield

    async def process_tiles(self, tiles_df):
        fut = self.client.submit(dask_process_tiles, self.worker_args_future, tiles_df)
        out, runtime_simulating = await asyncio.to_thread(fut.result)
        return out, runtime_simulating
