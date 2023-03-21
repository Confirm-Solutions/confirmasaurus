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
step_id = 2
n_zones = 4
```

```python
self = db
coordination_id = self.con.query(
    """
    select max(coordination_id) + 1 from results
    """
).fetchone()[0]
```

```python
ordering = ",".join(
    [f"theta{i}" for i in range(self.dimension())]
    + [c for c in self._results_columns() if c.startswith("null_truth")]
)
ordering
```

```python
%%time
import numpy as np
id_df = self.con.query(f'''
        select id from results
                where eligible=true
                    and active=true
                 order by {ordering}
''').df()
id_df['zone_id'] = np.arange(id_df.shape[0]) % n_zones
```

```python
%%time
self.con.query(f'''
    insert into zone_mapping
        select id_df.id, {coordination_id}, results.zone_id, id_df.zone_id, {step_id}
            from id_df
            left join results using (id)
''')
```

```python
%%time
self.con.execute(
    f"""
    update results set 
            zone_id=(
                select new_zone_id from zone_mapping
                    where zone_mapping.id=results.id
                    and zone_mapping.coordination_id={coordination_id}
            ),
            coordination_id={coordination_id}
        where eligible=true
            and active=true
"""
)
```

```python
n_results = self.con.query(
    f""" 
    select count(*) from results where coordination_id={coordination_id}
"""
).fetchone()[0]
zone_steps = dict(
    self.con.query(
        f"""
    select zone_id, max(step_id) from results
            where coordination_id={coordination_id}
        group by zone_id
        order by zone_id
"""
    ).fetchall()
)
```

```python
reports_df = ch.query_df(client, 'select * from reports')
```
