```python
import time

start = time.time()
import numpy as np  # noqa: E402
import imprint as ip  # noqa: E402
ip.setup_nb()

from imprint.models.ztest import ZTest1D  # noqa: E402
import confirm.adagrid as ada  # noqa: E402
from confirm.cloud.modal_backend import ModalBackend  # noqa: E402
import confirm.models.wd41 as wd41

import dotenv

dotenv.load_dotenv()
# d = 3
# g = ip.cartesian_grid(
#     theta_min=np.full(d, -1),
#     theta_max=np.full(d, 0),
#     null_hypos=[ip.hypo("theta0 > theta1")],
# )
model = wd41.WD41(0, 1, ignore_intersection=True)
g = ip.cartesian_grid(
    [-2.5, -2.5, -2.5, -2.5],
    [1.0, 1.0, 1.0, 1.0],
    n=[10, 10, 10, 10],
    null_hypos=model.null_hypos,
)
db = ada.ada_validate(
    wd41.WD41,
    model_kwargs={"ignore_intersection": True},
    g=g,
    lam=-1.96,
    prod=False,
    n_K_double=7,
    max_target=0.001,
    global_target=0.002,
    step_size=2**17,
    packet_size=2**13,
    n_steps=6,
    n_zones=4,
    coordinate_every=2,
    # backend=CoiledBackend()
    backend=ModalBackend(n_workers=1, gpu="any"),
)
```

```python
report_df = db.get_reports()
report_df['status']
```

```python
# skip the first step
first_non_work = (report_df['status'] == 'WORKING').argmin()

start = report_df.iloc[first_non_work]['time']
end = report_df.iloc[-2]['time']
runtime_total = end - start
included = report_df.iloc[first_non_work:]
included_work = included[included['status'] == 'WORKING']
included_nonwork = included[included['status'] != 'WORKING']
runtime_sim = included_work['runtime_simulating'].sum()
runtime_process_tiles = included_work['runtime_process_tiles'].sum()
runtime_step_coord = included_nonwork['runtime_total'].sum()
runtime_sim, runtime_process_tiles, runtime_step_coord, runtime_total
```

```python
100 * runtime_sim / runtime_total
```
