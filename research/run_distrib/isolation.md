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
```

### Glossary

- **coordination**: All workers come together and divide up all tiles.
- **step**: A worker checks the convergence criterion, select tiles, and simulates.
- **packet**: The unit of simulation work that a worker requests from the DB.
- **batch**: The unit of simulation work that is passed to a Model.

```python
get_ipython().__class__.__name__
```

```python
get_ipython().config
```

```python
asyncio.get_running_loop()
```

```python
ch.clear_dbs(drop_all_redis_keys=True)
```

```python
db = ch.Clickhouse.connect()
g = ip.cartesian_grid(
    theta_min=[-1], theta_max=[1], n=[50], null_hypos=[ip.hypo("x0 < 0")]
)
ada_validate(ZTest1D, lam=-1.96, g=g, db=db, prod=False)
```

```python
ada.db.get_tiles()
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
