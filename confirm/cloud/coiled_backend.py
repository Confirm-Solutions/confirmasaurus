import asyncio
import contextlib
import logging
import os
import subprocess

import dask
import jax
from distributed import get_worker
from distributed.diagnostics.plugin import UploadDirectory

import confirm
import imprint as ip
from ..adagrid.adagrid import LocalBackend

logger = logging.getLogger(__name__)


def create_software_env():
    reqs = subprocess.run(
        "poetry export --with=cloud --without-hashes",
        stdout=subprocess.PIPE,
        shell=True,
    ).stdout.decode("utf-8")
    reqs = reqs.split("\n")

    req_jax = [r.split(";")[0][:-1] for r in reqs if "jax==" in r][0].split("==")[1]
    reqs = [
        r.split(";")[0][:-1]
        for r in reqs
        if ("jax" not in r and 'sys_platform == "win32"' not in r)
    ]
    reqs.append(
        "--find-links "
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    reqs.append(f"jax[cuda11_pip]=={req_jax}")
    reqs = [r for r in reqs if r != ""]
    name = "confirm-coiled"
    import coiled

    coiled.create_software_environment(name=name, pip=reqs)
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


def setup_cluster(update_software_env=True, n_workers=1, idle_timeout="60 minutes"):
    if update_software_env:
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
        allow_ssh=True,
        wait_for_workers=1,
        worker_options={"nthreads": 2},
    )
    cluster.scale(n_workers)
    client = cluster.get_client()
    reset_confirm_imprint(client)
    return cluster, client


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

    (
        algo_type,
        model_type,
        model_args,
        model_kwargs,
        null_hypos,
        cfg,
        environ,
    ) = worker_args
    # Hashing the arguments lets us avoid re-initializing the model and algo
    # if the arguments are the same.
    # NOTE: this could be extended to use an `algo` dictionary, where the key
    # is this hash. But, I'm suspicious that would cause out-of-memory errors
    # so the design is currently limited to a single algo per worker at a
    # single point in time.
    hash_args = hash(
        (
            algo_type.__name__,
            model_type.__name__,
            model_args,
            tuple(model_kwargs.items()),
            tuple([h.description() for h in null_hypos]),
            tuple(cfg.items()),
            tuple(environ.items()),
        )
    )
    has_hash = hasattr(worker, "algo_hash")
    has_algo = hasattr(worker, "algo")
    if not (has_algo and has_hash) or (has_hash and hash_args != worker.algo_hash):
        ip.package_settings()

        # Need to insert variables into the environment for Clickhouse:
        #
        # NOTE: this is probably a little bit insecure and ideally we would
        # transfer the encrypted secrets file and then use sops to decrypt it
        # here. But that would probably require redoing the coiled software
        # environment to use a docker image and I don't want to deal with that
        # now.
        os.environ.update(environ)
        import confirm.cloud.clickhouse as ch

        db = ch.ClickhouseTiles.connect(
            job_name=cfg["job_name"], service=cfg["clickhouse_service"]
        )

        model = model_type(*model_args, **model_kwargs)
        worker.algo = algo_type(model, null_hypos, db, cfg, None)
        worker.algo_hash = hash_args


def dask_process_tiles(worker_args, tiles_df, refine_deepen):
    setup_worker(worker_args)
    jax_platform = jax.lib.xla_bridge.get_backend().platform
    assert jax_platform == "gpu"

    worker = get_worker()
    lb = LocalBackend()
    lb.algo = worker.algo
    return lb.sync_submit_tiles(tiles_df, refine_deepen)


class CoiledBackend(LocalBackend):
    def __init__(
        self,
        update_software_env: bool = True,
        n_workers: int = 1,
        detach: bool = False,
        restart_workers: bool = False,
        client=None,
    ):
        super().__init__()
        self.use_clickhouse = True
        self.update_software_env = update_software_env
        self.restart_workers = restart_workers
        self.detach = detach
        self.n_workers = n_workers
        self.client = client

    def check(self):
        return check().compute()

    def get_cfg(self):
        out = super().get_cfg()
        out.update(
            {
                "update_software_env": self.update_software_env,
                "restart_workers": self.restart_workers,
                "detach": self.detach,
                "n_workers": self.n_workers,
            }
        )
        return out

    @contextlib.asynccontextmanager
    async def setup(self, algo):
        with contextlib.ExitStack() as stack:
            if self.client is None:
                self.cluster, self.client = setup_cluster(
                    update_software_env=self.update_software_env,
                    n_workers=self.n_workers,
                )
                stack.enter_context(contextlib.closing(self.cluster))
                stack.enter_context(contextlib.closing(self.client))
            if self.restart_workers:
                self.client.restart()
            filtered_cfg = {
                k: v for k, v in algo.cfg.items() if k in self.algo_cfg_entries
            }
            worker_args = (
                type(algo),
                type(algo.driver.model),
                (algo.cfg["model_seed"], algo.max_K),
                algo.cfg["model_kwargs"],
                algo.null_hypos,
                filtered_cfg,
                {k: v for k, v in os.environ.items() if k.startswith("CLICKHOUSE")},
            )
            self.worker_args_future = self.client.scatter(worker_args, broadcast=True)
            yield

    async def submit_tiles(self, tiles_df, refine_deepen):
        step_id = int(tiles_df["step_id"].iloc[0])
        # negative step_id is the priority so that earlier steps are completed
        # before later steps when we are using n_parallel_steps
        with dask.annotate(priority=-step_id):
            return self.client.submit(
                dask_process_tiles,
                self.worker_args_future,
                tiles_df,
                refine_deepen,
                retries=0,
            )

    async def wait_for_results(self, awaitable):
        return await asyncio.to_thread(awaitable.result)
