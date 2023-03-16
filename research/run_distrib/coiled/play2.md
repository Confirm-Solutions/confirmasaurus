```python
import imprint as ip
ip.setup_nb()
```

```python
from confirm.cloud.coiled_backend import CoiledBackend
import confirm.adagrid as ada
from imprint.models.ztest import ZTest1D

backend = CoiledBackend()
```

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db = ada.ada_calibrate(
    ZTest1D, g=g, nB=5, tile_batch_size=1, prod=True, n_zones=1, backend=backend
)
```

```python
db
```
