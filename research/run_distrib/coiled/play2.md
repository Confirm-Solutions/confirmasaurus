```python
import imprint as ip
ip.setup_nb()
```

```python
from confirm.cloud.coiled_backend import CoiledBackend, setup_cluster
import confirm.adagrid as ada
from imprint.models.ztest import ZTest1D

cluster = setup_cluster(idle_timeout="2 hours")
```

```python
backend = CoiledBackend(cluster=cluster)
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db = ada.ada_calibrate(
    ZTest1D,
    g=g,
    nB=5,
    prod=True,
    n_zones=1,
    backend=CoiledBackend(cluster=cluster),
)

```

```python
import numpy as np
d = 3
g = ip.cartesian_grid(
    theta_min=np.full(d, -1),
    theta_max=np.full(d, 0),
    null_hypos=[ip.hypo("theta0 > theta1")],
)
db = ada.ada_validate(
    ZTest1D,
    g=g,
    lam=-1.96,
    prod=False,
    n_K_double=7,
    max_target=0.001,
    global_target=0.002,
    step_size=2**15,
    packet_size=2**12,
    n_steps=10,
    backend=CoiledBackend(cluster=cluster)
    # backend=ModalBackend(n_workers=1, gpu="any"),
)
```

```python
tdf = db.con.query('select * from tiles where step_id=4 and packet_id=1').df()
tdf = tdf[tdf['K'] < 30000]
tdf.shape
```

```python
import confirm.cloud.coiled_backend as coiled_backend
client = await cluster.get_client()
coiled_backend.reset_confirm_imprint(client)
```

```python
from confirm.adagrid.validate import AdaValidate
worker_args_fut = client.scatter((
    ZTest1D,
    (0, 2**14),
    dict(),
    AdaValidate,
    dict(
        lam=-1.96,
        tile_batch_size=64,
        delta=0.01,
        global_target = 0.001, 
        max_target = 0.001, 
        init_K=2**12,
        n_K_double=4,
    ),
), broadcast=True)
```

```python
client.submit(coiled_backend.dask_process_tiles, worker_args_fut, tdf).result()
```
