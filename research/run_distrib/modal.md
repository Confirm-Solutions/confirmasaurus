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
    bias_target=0.0001,
)
```

```python
db = ch.Clickhouse.connect(job_id="distributed")
iter, reports, db = ada.ada_calibrate(
    ZTest1D,
    g=g,
    nB=5,
    packet_size=32,
    iter_size=8,
    grid_target=0.0001,
    bias_target=0.0001,
)
```

```python
tiles = db.get_all()
```

```python
ch.clear_dbs(db1.client, names=['distributed'], yes=True)
```

```python
# from confirm.adagrid.calibration import AdaCalibrationDriver, CalibrationConfig
# import json
# gtemp = ip.Grid(db1.get_all())
# null_hypos = [ip.hypo("x0 < 0")]
# c= CalibrationConfig(ZTest1D, *[None] * 16, defaults=db1.store.get('config').iloc[0].to_dict())
# model = ZTest1D(
#     seed=c.model_seed,
#     max_K=c.init_K * 2**c.n_K_double,
#     **json.loads(c.model_kwargs),
# )
# driver = AdaCalibrationDriver(None, model, null_hypos, c)
# driver.bootstrap_calibrate(gtemp.df, 0.025)
# gtemp.df['K'].value_counts()
```

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db = ch.Clickhouse.connect(job_id='distributed')
_ = ada_calibrate(ZTest1D, db=db, g=g, n_iter=1, iter_size=8, grid_target=0.0001, bias_target=0.0001)
```

```python
job_id = db.job_id
def worker(i):
    db = ch.Clickhouse.connect(job_id=job_id)
    return ada_calibrate(ZTest1D, db=db, n_iter=100)[:2]
# iter_two2, reports_two2, ada_two2 = step(0)
import modal
stub = modal.Stub("two_workers")
wrapper = modal_util.modalize(stub)(worker)
with stub.run():
    results = list(wrapper.map([1, 2]))
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
