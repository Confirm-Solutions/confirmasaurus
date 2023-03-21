```python
import confirm.cloud.clickhouse as ch
client = ch.connect('wd41_4d_v0')
ch.list_tables(client)
```

```python
import confirm.adagrid as ada
db = ada.DuckDBTiles.connect('wd41_4d_v7.db')
```

```python
db.con.query('select rowid from tiles where rowid > 2005 limit 10')
```

```python
reports_df = ch.query_df(client, 'select * from reports')
```
