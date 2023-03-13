```python
from imprint.nb_util import setup_nb
setup_nb()
```

```python
import os
import coiled
import imprint 
import confirm
from distributed.diagnostics.plugin import UploadDirectory

coiled.create_software_environment(name="confirm-coiled", conda="env-coiled.yml")
cluster = coiled.Cluster(
    name="confirm-coiled",
    software="confirm-coiled",
    n_workers=1,
    worker_vm_types=["g4dn.xlarge"],
    worker_gpu=1,
    compute_purchase_option='spot_with_fallback',
    shutdown_on_close=False
)
client = cluster.get_client()
```

```python
def upload_pkg(client, module, restart=False):
    from pathlib import Path

    dir = Path(module.__file__).parent.parent
    skip = [p for p in os.listdir(dir) if p != module.__name__]
    return client.register_worker_plugin(
        UploadDirectory(dir, skip_words=skip, restart=restart, update_path=True),
        nanny=True,
    )

upload_pkg(client, confirm)
upload_pkg(client, imprint, restart=True)
```

```python

import dask
@dask.delayed
def check_nvidia_jax():
    import jax
    import subprocess
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    nvidia_smi = result.stdout.decode('ascii')
    jax_platform = jax.lib.xla_bridge.get_backend().platform
    key = jax.random.PRNGKey(0)
    means = jax.random.uniform(key, (10000, 2)).mean(axis=0)
    return nvidia_smi, jax_platform, means

nvidia_smi, jax_platform, means = check_nvidia_jax().compute()
print(nvidia_smi)
print(jax_platform, means)

@dask.delayed
def check_confirm_imprint():
    import confirm
    import imprint
check_confirm_imprint().compute()
```

```python
def setup_worker(worker_args):
    import imprint as ip
    from distributed import get_worker
    ip.package_settings()
    worker = get_worker()
    (model_type, model_args, model_kwargs, algo_type, cfg) = worker_args
    hash_args = hash((
        model_type.__name__,
        model_args,
        tuple(model_kwargs.items()),
        algo_type.__name__,
        tuple(cfg.items()),
    ))
    has_hash = hasattr(worker, 'algo_hash')
    has_algo = hasattr(worker, 'algo')
    if not (has_algo and has_hash) or (has_hash and hash_args != worker.algo_hash):
        model = model_type(*model_args, **model_kwargs)
        cfg['worker_id'] = 2
        worker.algo = algo_type(model, None, None, cfg, None) 
        worker.algo_hash = hash_args
    async def async_process_tiles(tiles_df):
        return await worker.algo.process_tiles(tiles_df = tiles_df)
    import synchronicity
    synchronizer = synchronicity.Synchronizer()
    process_tiles = synchronizer.create(async_process_tiles)[synchronicity.Interface.BLOCKING]
    return process_tiles

@dask.delayed
def dask_process_packet(worker_args, packet_df):
    import jax
    import time
    start = time.time()
    process_tiles = setup_worker(worker_args)
    jax_platform = jax.lib.xla_bridge.get_backend().platform
    assert jax_platform == 'gpu'
    runtime_non_process = time.time() - start
    start = time.time()
    out = process_tiles(packet_df)
    runtime = time.time() - start
    return out, runtime, runtime_non_process

import time
import imprint as ip
import dask
import numpy as np
from confirm.adagrid.calibrate import AdaCalibrate
from confirm.models.wd41 import WD41
from imprint.models.ztest import ZTest1D
NN = 6
g = ip.cartesian_grid(
    theta_min=np.full(4, -1),
    theta_max=np.full(4, 0),
    n=np.full(4, NN),
    null_hypos=[ip.hypo("theta0 > theta1")],
)
init_K = 2**13
K = init_K * 16
g.df['K'] = init_K
cfg = dict(
    init_K = 2**13,
    n_K_double = 4,
    tile_batch_size = 64,
    # lam = -1.96, 
    # delta = 0.01,
    # global_target = 0.001,
    # max_target = 0.01,
    bootstrap_seed=0,
    nB=50,
    alpha=0.025,
    calibration_min_idx=40
)
```

```python
# worker_args = (WD41, (0, K), dict(ignore_intersection=True), AdaCalibrate, cfg)
worker_args = (ZTest1D, (0, K), dict(), AdaCalibrate, cfg)
worker_args_future = client.scatter(worker_args, broadcast=True)
start = time.time()
out, inner_runtime, ir = dask_process_packet(worker_args_future, g.df).compute()
runtime1 = time.time() - start
start = time.time()
out, inner_runtime2, ir2 = dask_process_packet(worker_args_future, g.df).compute()
runtime2 = time.time() - start
print(runtime1, inner_runtime, ir)
print(runtime2, inner_runtime2, ir2)
```

```python
12.3 / 200.
```

```python
import asyncio
def dpp():
    out, inner_runtime2, ir2 = dask_process_packet(worker_args_future, g.df).compute()
    return inner_runtime2

async def lots_test():
    ts = [
        asyncio.create_task(asyncio.to_thread(dpp))
        for i in range(1000)
    ]
    return await asyncio.gather(*ts)
    
start = time.time()
await lots_test()
print(time.time() - start)
```

```python
dpp()
```

```python
2.21 / (g.df.shape[0] * K) * 1e9
```

```python
import cloudpickle
```

```python
abcdef = cloudpickle.dumps(g.df)
```

```python
g.df.memory_usage().sum() / 1e6, out.memory_usage().sum() / 1e6
```

```python
import logging
logger = logging.getLogger('confirm.cloud.coiled')
class CoiledAlgo:
    def __init__(self, client, worker_args_future, wrapped):
        super().__init__()
        self.client = client
        self.wrapped = wrapped
        self.worker_args_future = worker_args_future

        self.null_hypos = wrapped.null_hypos
        self.cfg = wrapped.cfg
        self.db = wrapped.db
        self.callback = wrapped.callback
        self.Ks = wrapped.Ks
        self.max_K = wrapped.max_K
        self.driver = wrapped.driver
        
    def get_orderer(self):
        return self.wrapped.get_orderer()
        
    async def process_tiles(self, *, tiles_df):
        start = time.time()
        relevant_cols = [
            c for c in tiles_df.columns if c.startswith('theta') or c.startswith('K')
        ]
        out, compute_runtime = dask_process_packet(self.worker_args_future, tiles_df).compute()
        total_runtime = time.time() - start
        # logger.debug('Compute runtime: %s', compute_runtime)
        # logger.debug('Total runtime: %s', total_runtime)
        print('Compute runtime: %s', compute_runtime)
        print('Total runtime: %s', total_runtime)
        return out
        
    async def convergence_criterion(self, zone_id, report):
        return await self.wrapped.convergence_criterion(zone_id, report)

    async def select_tiles(self, zone_id, new_step_id, report, convergence_task):
        return await self.wrapped.select_tiles(zone_id, new_step_id, report, convergence_task)
```

```python
import contextlib
from confirm.adagrid.backend import LocalBackend
class CoiledBackend(LocalBackend):
    def __init__(self, client, n_zones=1, coordinate_every=5):
        super().__init__(n_zones, coordinate_every)
        self.client = client

    @contextlib.asynccontextmanager
    async def setup(self, algo_type, algo, kwargs):
        algo_entries = [
            'init_K',
            'n_K_double',
            'tile_batch_size',
            'lam',
            'delta',
            'worker_id',
            'global_target',
            'max_target',
            'bootstrap_seed',
            'nB',
            'alpha',
            'calibration_min_idx'
        ]
        filtered_cfg = {k: v for k, v in algo.cfg.items() if k in algo_entries}
        worker_args = (type(algo.driver.model), (algo.cfg['model_seed'], algo.max_K), algo.cfg['model_kwargs'], algo_type, filtered_cfg)
        worker_args_future = client.scatter(worker_args, broadcast=True)
        self.algo = CoiledAlgo(self.client, worker_args_future, algo)
        yield
```

```python
import numpy as np
import imprint as ip
from imprint.models.ztest import ZTest1D
import confirm.adagrid as ada

d = 2
g = ip.cartesian_grid(
    theta_min=np.full(d, -1),
    theta_max=np.full(d, 0),
    null_hypos=[ip.hypo("theta0 > theta1")],
)
db = ada.ada_calibrate(ZTest1D, g=g, alpha=0.025, prod=False, backend=CoiledBackend(client))
```

```python
import confirm.models.wd41 as wd41
model = wd41.WD41(0, 1, ignore_intersection=True)
grid = ip.cartesian_grid(
    [-2.5, -2.5, -2.5, -2.5],
    [1.0, 1.0, 1.0, 1.0],
    n=[10, 10, 10, 10],
    null_hypos=model.null_hypos,
)
db = ada.DuckDBTiles.connect('./wd41-4.db')
ada.ada_calibrate(
    wd41.WD41,
    g=grid,
    db=db,
    alpha=0.025,
    bias_target=0.0025,
    grid_target=0.0025,
    std_target=0.005,
    n_K_double=6,
    calibration_min_idx=100,
    step_size=2**18,
    packet_size=2**15,
    model_kwargs={"ignore_intersection": True},
    backend=CoiledBackend(client)
)
```

```python
client.close()
cluster.close()
```

```python
import dask

@dask.delayed
def run_ada():
    import numpy as np
    import imprint as ip
    from imprint.models.ztest import ZTest1D
    import confirm.adagrid as ada
    ip.package_settings()
    d = 2
    g = ip.cartesian_grid(
        theta_min=np.full(d, -1),
        theta_max=np.full(d, 0),
        null_hypos=[ip.hypo("theta0 > theta1")],
    )
    db = ada.ada_validate(
        ZTest1D,
        g=g,
        lam=-1.96,
        prod=False
    )
    return ip.Grid(db.get_results(), 1).prune_inactive().n_tiles

NN = run_ada().compute()
NN
```
