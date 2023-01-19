```python
import imprint.nb_util as nb_util
nb_util.setup_nb()
import os
import time
import imprint as ip
from imprint.models.ztest import ZTest1D
import confirm.adagrid as ada
import confirm.cloud.clickhouse as ch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
iter, reports, db = ada.ada_calibrate(
    ZTest1D,
    g=g,
    nB=5,
    tile_batch_size=1,
    packet_size=32,
    iter_size=32,
    grid_target=0.0001,
    bias_target=0.0002,
)
```

```python
db_ch = ch.Clickhouse.connect()
iter, reports, _ = ada.ada_calibrate(
    ZTest1D,
    g=g,
    db=db_ch,
    nB=5,
    packet_size=32,
    iter_size=8,
    grid_target=0.0001,
    bias_target=0.0002,
)
```

```python
ch.clear_dbs(ch.get_ch_client(), names=['distributed'], yes=True)
```

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db_dist = ch.Clickhouse.connect(job_id="distributed")
iter, reports, _ = ada.ada_calibrate(
    ZTest1D,
    g=g,
    db=db_dist,
    nB=5,
    n_iter=0,
    packet_size=32,
    iter_size=8,
    grid_target=0.0001,
    bias_target=0.0002,
)
db_w = ch.Clickhouse.connect(job_id="distributed")
iter, reports, _ = ada.ada_calibrate(ZTest1D, db=db_w, n_iter=100)
```

## Run two threaded workers

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db_dist = ch.Clickhouse.connect(job_id="threaded")
iter, reports, _ = ada.ada_calibrate(
    ZTest1D,
    g=g,
    db=db_dist,
    nB=5,
    n_iter=0,
    packet_size=32,
    iter_size=8,
    grid_target=0.0001,
    bias_target=0.0002,
)
```

```python
from concurrent.futures import ThreadPoolExecutor, wait
def worker():
    import confirm.adagrid as ada
    import confirm.cloud.clickhouse as ch
    from imprint.models.ztest import ZTest1D
    db_w = ch.Clickhouse.connect(job_id='threaded')
    return ada.ada_calibrate(ZTest1D, db=db_w, n_iter=100)[:2]

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(worker) for i in range(2)]
    wait(futures)
    outputs = [f.result() for f in futures]
```

## Run two distributed workers

```python
ch.clear_dbs(ch.get_ch_client(), names=['distributed'], yes=True)
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db_dist = ch.Clickhouse.connect(job_id="distributed")
iter, reports, _ = ada.ada_calibrate(
    ZTest1D,
    g=g,
    db=db_dist,
    nB=5,
    n_iter=0,
    packet_size=32,
    iter_size=8,
    grid_target=0.0001,
    bias_target=0.0002,
)
```

```python

import confirm.cloud.modal_util as modal_util
import modal
stub = modal.Stub("two_workers")

@stub.function(
    image=modal_util.get_image(dependency_groups=["test", "cloud"]),
    retries=0,
    mounts=modal.create_package_mounts(["confirm", "imprint"]),
    secret=modal.Secret.from_name("confirm-secrets")
)
def worker(i):
    import confirm.adagrid as ada
    import confirm.cloud.clickhouse as ch
    from imprint.models.ztest import ZTest1D
    db_w = ch.Clickhouse.connect(job_id='distributed')
    return ada.ada_calibrate(ZTest1D, db=db_w, n_iter=100)[:2]

with stub.run():
    results = list(worker.map([1, 2]))
```

```python
modal.__version__
```

```python
duckdb.__path__
```

```python
# With this import, error! Without the error, fine.
# import duckdb
import modal
stub = modal.Stub("error")

@stub.function()
def worker(i):
    return i + 1

with stub.run():
    print(worker.call(0))
```

```python
db.worst_tile('lams')
```

```python
df2 = db.get_all()
plt.plot(np.sort(df2['theta0']))
plt.show()
plt.plot(df2['theta0'], df2['lams'], 'o')
plt.show()
```

```python
plt.plot(np.sort(db1.get_all()['theta0']))
plt.show()
```

```python

```
