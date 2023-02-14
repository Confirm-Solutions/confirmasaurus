```python
from imprint.nb_util import setup_nb
setup_nb()
import asyncio
import redis
import numpy as np

import confirm.cloud.clickhouse as ch

from confirm.adagrid.validate import AdaValidate
from confirm.adagrid.calibrate import AdaCalibrate
from confirm.adagrid.validate import ada_validate
from confirm.adagrid.calibrate import ada_calibrate
from confirm.cloud.redis_heartbeat import HeartbeatThread
import confirm.adagrid.adagrid as adagrid
import imprint as ip
from imprint.models.ztest import ZTest1D

import clickhouse_connect
clickhouse_connect.common.set_setting('autogenerate_session_id', False)
```

### Glossary

- **coordination**: All workers come together and divide up all tiles.
- **step**: A worker checks the convergence criterion, select tiles, and simulates.
- **packet**: The unit of simulation work that a worker requests from the DB.
- **batch**: The unit of simulation work that is passed to a Model.

```python
ch.clear_dbs(drop_all_redis_keys=True)
```

```python
db = ch.Clickhouse.connect(job_id="8603b7cfab1445bc9a3ebfcfcd956255")
```

```python
tiles = db.get_tiles()
```

```python
tiles
```

```python
db.get_work(0, 2, 2, 3)
```

```python
import time
start = time.time()
db = ch.Clickhouse.connect()
print('1', time.time() - start)
g = ip.cartesian_grid(
    theta_min=[-1], theta_max=[1], n=[50000], null_hypos=[ip.hypo("x0 < 0")]
)
print('2', time.time() - start)

import inspect
sig = inspect.signature(ada_validate)
kwargs={k: v.default for k,v in sig.parameters.items()}
kwargs.update(dict(lam=-1.96, prod=False))
n_workers=2
ada = adagrid.Adagrid()
print('3', time.time() - start)
await ada.init(ZTest1D, g, db, AdaValidate, print, None, n_workers, kwargs, worker_id = 1)
print('4', time.time() - start)
```

```python
ada.db.get_tiles()
```

```python
ada.db.get_results()
```

```python
ada.db.get_work(0, 1, 3, 0)
```

```python
df = ada.db.get_results()
np.sum(df['step_id'] == 3)
```

```python
ada.db.n_processed_tiles(1, 3)
```

```python
async with ada.db.heartbeat(ada.worker_id):
    await ada._run_local()
```

```python
async with HeartbeatThread(ada.db.redis_con, ada.db.job_id, ada.worker_id):
    await ada.process_step(0, 0)
```

```python
db.get_reports()
```

```python
await ada.new_step(1)
```

```python
# TODO: test running coordinate with a single work and confirming that it
# updates eligible and active.
pre = ch._query_df(db.client, 'select * from results')
pre['active'].sum(), pre['eligible'].sum()
```

```python
async with HeartbeatThread(ada.db.redis_con, ada.db.job_id, ada.worker_id):
    status = await ada.coordinate()
```

```python
ch._query_df(db.client, 'select * from results')['active'].sum()
```

```python
ch._query_df(db.client, 'select * from results where coordination_id = 1')
```

```python
ada.db.get_tiles()
```

```python
await ada.process_step(1)
```
