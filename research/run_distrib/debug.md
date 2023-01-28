```python
import confirm.cloud.clickhouse as ch

# Bdb = ch.Clickhouse.connect(job_id = '544d8afac72e4633809a162a480ed998')
Adb = ch.Clickhouse.connect(job_id="41ef6dbb374d4bb9a62a0ea8d98b5c9a")
Bdb = ch.Clickhouse.connect(job_id="0da447c75edf4ef7b2cc905c6c004952")
```

```python
At = Adb.get_results()
Bt = Bdb.get_results()
```

```python
At.shape, Bt.shape
```

```python
import pandas as pd

drop_cols = ["id", "parent_id", "step_iter", "creator_id", "processor_id", "creation_time", "processing_time"]

AAA = At.drop(drop_cols, axis=1).sort_values(['step_id', 'theta0']).reset_index(drop=True)
BBB = Bt.drop(drop_cols, axis=1).sort_values(['step_id', 'theta0']).reset_index(drop=True)
pd.testing.assert_frame_equal(AAA, BBB, check_dtype=False)
```

```python
AAA['twb_max_lams'] - BBB['twb_max_lams']
```

```python
pd.set_option('display.max_columns', None)
display(AAA.head(2)), display(BBB.head(2))
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
