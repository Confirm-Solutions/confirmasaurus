```python
from imprint.nb_util import setup_nb
setup_nb()
import confirm.adagrid as ada
import imprint as ip
from imprint.models.ztest import ZTest1D

g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
db = ada.ada_calibrate(
    ZTest1D,
    g=g,
    init_K=1024,
    nB=2,
    grid_target=0.005,
    bias_target=0.005,
    std_target=0.005,
    prod=False,
    tile_batch_size=1,
    coordinate_every=1,
    n_zones=2
)
```

```python
df = db.get_results()
```

```python
import confirm.cloud.clickhouse as ch
chdb = ch.Clickhouse.connect()
```

```python
import logging

logger = logging.getLogger("imprint")

ch.backup(db, chdb)
```

```python

db2 = ada.DuckDBTiles.connect()
ch.restore(db2, chdb)
```

```python
import pandas as pd
for table in ch.all_tables:
    if not db.does_table_exist(table):
        continue
    orig = db.con.query(f'select * from {table}').df()
    restored = db2.con.query(f'select * from {table}').df()
    pd.testing.assert_frame_equal(orig, restored)
```

```python
log_df = db.con.query('select * from logs').df()
log_df.head()['t']
```

```python
log_df2 = db2.con.query('select * from logs').df()
log_df2.head()['t']
```

```python

```
