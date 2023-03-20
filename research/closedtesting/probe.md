```python
import confirm.cloud.clickhouse as ch
client = ch.connect('wd41_4d_v0')
ch.list_tables(client)
```

```python
reports_df = ch.query_df(client, 'select * from reports')
```
