```python
import confirm.cloud.clickhouse as ch
client = ch.get_ch_client()
ch.clear_dbs(client)
```

```python
import time
start = time.time()

import os
import imprint as ip
from imprint.models.ztest import ZTest1D
from confirm.adagrid.calibration import ada_calibrate
import confirm.cloud.clickhouse as ch
import pandas as pd

print("Loaded confirm in {:.2f} seconds".format(time.time() - start))
start = time.time()
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
print("Created grid in {:.2f} seconds".format(time.time() - start))
start = time.time()
db = ch.Clickhouse.connect()
iter, reports, ada = ada_calibrate(ZTest1D, g=g, db=db, nB=5, tile_batch_size=1)
print("Ran ada in {:.2f} seconds".format(time.time() - start))
# print(pd.DataFrame(reports))
```

```python
import confirm.cloud.modal_util as modal_util
import modal

img = modal_util.get_image()
```

```python
stub = modal.Stub("e2e_runner")
@stub.function(
    image=img,
    # gpu=True,
    gpu=modal.gpu.A100(),
    retries=0,
    mounts=modal.create_package_mounts(["confirm"]),
)
def worker():
    pass
```
