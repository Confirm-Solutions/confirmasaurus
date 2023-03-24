```python
import duckdb
con = duckdb.connect()
```

```python
import pandas as pd
import numpy as np
df = pd.DataFrame({
    "id": np.arange(25),
    "coordination_id": np.zeros(25, dtype=np.int32),
    "step_id": np.zeros(25, dtype=np.int32),
    "zone_id": np.arange(25)
})
con.execute("CREATE OR REPLACE TABLE test as select * from df")
```

```python
con.execute('create or replace table mapping (id int, coordination_id int, old_zone_id int, new_zone_id int, before_step_id int)')
```

```python
coordination_id = 1
step_id = 5
n_zones = 4
con.execute(f"""
insert into mapping 
    select id, {coordination_id}, zone_id,
        (row_number() OVER ())%{n_zones},
        {step_id}
        from test
"""
)   
con.execute(f"""
update test set 
    zone_id=(
        select new_zone_id from mapping
            where mapping.id=test.id
            and mapping.coordination_id={coordination_id}
    ),
    coordination_id={coordination_id}
""")
```

```python
con.query('select * from mapping')
```

```python
con.query('select * from test')
```

```python
import imprint as ip
ip.setup_nb()
```

```python
import confirm.cloud.coiled_backend as coiled_backend
import confirm.adagrid as ada
from imprint.models.ztest import ZTest1D

cluster = coiled_backend.setup_cluster(n_workers=4, idle_timeout="2 hours")
client = cluster.get_client()
```

```python
coiled_backend.reset_confirm_imprint(client)
```

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db = ada.ada_calibrate(
    ZTest1D,
    g=g,
    nB=5,
    prod=True,
    n_zones=1,
    backend=coiled_backend.CoiledBackend(cluster=cluster),
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
