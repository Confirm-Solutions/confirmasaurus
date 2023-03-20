```python
# runtime duckdb = 3.8978
# runtime clickhouse = 37.15
# runtime ch + modal = 57.5
```

```python
from imprint.nb_util import setup_nb
setup_nb()
from confirm.adagrid.debug import get_logs

#modal + ch
# logs, log_df, db = get_logs('44dcf9061f5f4bcd81d85255f35cdedc')
# just ch
# logs, log_df, db = get_logs('a73de730d75a4b248974dac8bd7aeba2')

# big ch
logs, log_df, db = get_logs('791ead8ad7ae463e8a98d730966ec15b')
```

```python
log_df[(log_df['name'] == 'confirm.adagrid.step') & (log_df['message'].str.contains('packet insertion'))]
```

```python
backend_times = log_df[(log_df['name'] == 'confirm.adagrid.backend') & (log_df['message'].str.contains('took'))]['message']

split = backend_times.str.split('took')
split.str[1].astype(float).sum()
```

```python
backend_times
```

```python
rpt_df = db.get_reports()
```

```python
rpt_df[['worker_id', 'zone_id', 'step_id', 'n_tiles', 'status', 'runtime_total', 'runtime_process_tiles']]
```

```python
rpt_df['runtime_total'].sum()
```

```python
rpt_df['runtime_process_tiles'].sum()
```

```python
rpt_df.query('status == "WORKING" and step_id == 12')['runtime_process_tiles'].sum()
```
