import asyncio
import contextlib
import logging
import os
import subprocess

import dask
import distributed
import jax
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


def setup_cluster(
    update_software_env=True, n_workers=1, scale: bool = True, idle_timeout="60 minutes"
):
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
    if scale:
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


def setup_worker(dask_worker, worker_args=None):
    import os

    os.system("pip install line_profiler")
    (
        algo_type,
        model_type,
        cfg,
        environ,
    ) = worker_args

    ip.package_settings()

    # Need to insert variables into the environment for Clickhouse:
    #
    # NOTE: this is probably a little bit insecure and ideally we would
    # transfer the encrypted secrets file and then use sops to decrypt it
    # here. But that would probably require redoing the coiled software
    # environment to use a docker image and I don't want to deal with that
    # now.
    # Coiled also supports including secrets in the software environment or
    # Cluster configuration.
    os.environ.update(environ)

    jax_platform = jax.lib.xla_bridge.get_backend().platform
    assert jax_platform == "gpu"

    import confirm.cloud.clickhouse as ch

    db = ch.ClickhouseTiles.connect(
        job_name=cfg["job_name"], service=cfg["clickhouse_service"]
    )
    logger.debug("Connected to Clickhouse")

    dask_worker.algo = algo_type(model_type, db, cfg, None)
    assert dask_worker.algo.driver is not None


def raise_db_exceptions(dask_worker):
    if hasattr(dask_worker, "algo"):
        dask_worker.algo.db.cleanup()


def dask_process_tiles(tiles_df, refine_deepen, report):
    if ip.grid.worker_id == 1:
        client = distributed.get_client()
        worker_id = distributed.Queue("worker_id", client=client).get()
        logger.debug("Got worker_id: %s", worker_id)
        ip.grid.worker_id = worker_id

    dask_worker = distributed.get_worker()
    # Little hack... wait for the setup_worker function to run on the worker.
    # This shouldn't be necessary though!
    for i in range(100):
        if hasattr(dask_worker, "algo"):
            break
        else:
            import time

            time.sleep(0.1)

    lb = LocalBackend()
    lb.algo = dask_worker.algo
    out = lb.sync_submit_tiles(tiles_df, refine_deepen, report)
    raise_db_exceptions(dask_worker)
    return out


class CoiledBackend(LocalBackend):
    def __init__(
        self,
        update_software_env: bool = True,
        n_workers: int = 1,
        scale: bool = True,
        detach: bool = False,
        restart_workers: bool = False,
        client=None,
    ):
        super().__init__()
        self.use_clickhouse = True
        self.update_software_env = update_software_env
        self.restart_workers = restart_workers
        self.scale = scale
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
                    scale=self.scale,
                )
                stack.enter_context(contextlib.closing(self.cluster))
                stack.enter_context(contextlib.closing(self.client))
            if self.restart_workers:
                self.client.restart()
            filtered_cfg = {
                k: v for k, v in algo.cfg.items() if k in self.algo_cfg_entries
            }
            self.queue = distributed.Queue("worker_id", client=self.client)
            for i in range(2, 2 + 5 * self.n_workers):
                self.queue.put(i)
            worker_args = (
                type(algo),
                algo.model_type,
                filtered_cfg,
                {k: v for k, v in os.environ.items() if k.startswith("CLICKHOUSE")},
            )
            self.setup_task = asyncio.create_task(
                asyncio.to_thread(
                    self.client.run, setup_worker, worker_args=worker_args
                )
            )
            yield
            self.client.run(raise_db_exceptions)

    async def submit_tiles(self, tiles_df, refine_deepen, report):
        await self.setup_task
        step_id = int(tiles_df["step_id"].iloc[0])
        # negative step_id is the priority so that earlier steps are completed
        # before later steps when we are using n_parallel_steps
        with dask.annotate(priority=-step_id):
            return self.client.submit(
                dask_process_tiles,
                tiles_df,
                refine_deepen,
                report,
                retries=0,
            )

    async def wait_for_results(self, awaitable):
        return await asyncio.to_thread(awaitable.result)
