```python
db = ch.Clickhouse.connect(job_id = '544d8afac72e4633809a162a480ed998')
```

```python
T = db.get_all()
```

```python
T.loc[T['id'] == 4373889141130919936]
```

```python
from imprint.grid import _gen_short_uuids
_gen_short_uuids(n=2, worker_id=3, t=1674261200)
```

```python
import confirm.cloud.clickhouse as ch

Adb = ch.Clickhouse.connect(job_id = '1cf6481207e54adc842def1c2bb22dc7')
Bdb = ch.Clickhouse.connect(job_id = '544d8afac72e4633809a162a480ed998')
```

```python
_gen_short_uuids(n=2, worker_id=3, t=1674261200)
```

```python
At = Adb.get_all()
Bt = Bdb.get_all()
```

```python
At.shape, Bt.shape
```

```python
import pandas as pd

drop_cols = ["id", "parent_id", "step_iter", "worker_id"]
pd.testing.assert_frame_equal(
    At.drop(drop_cols, axis=1).sort_values(['step_id', 'theta0']).reset_index(drop=True), 
    Bt.drop(drop_cols, axis=1).sort_values(['step_id', 'theta0']).reset_index(drop=True)
)
```

```python
from IPython.display import display
```

```python
ch._query_df(Adb.client, "select id from tiles group by id having count(*) > 1")
```

```python
ch._query_df(Bdb.client, "select id from tiles group by id having count(*) > 1")
```

```python
ch._query_df(Bdb.client, "select id from results group by id having count(*) > 1")
```

```python
ch._query_df(Bdb.client, "select id from done group by id having count(*) > 1")
```

```python
ch._query_df(Bdb.client, 'select * from done where id = 4373889141130919936')
```

```python
ch._query_df(Bdb.client, 'select * from tiles where id = 4373889141130919936')
```

```python
ch._query_df(Bdb.client, 'select * from results where id = 4373889141130919936').columns
```

```python
for step_id in range(14):
    AA = At.loc[At['step_id'] == step_id]
    BB = Bt.loc[Bt['step_id'] == step_id]
    print(step_id, AA.shape, BB.shape)
    if AA.shape[0] != BB.shape[0]:
        display(AA)
        display(BB)
```

```python

```
