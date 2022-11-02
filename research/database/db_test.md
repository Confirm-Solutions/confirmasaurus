---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('confirm')
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
import sqlite3
import pickle

with open('../adagrid/4d/2090.pkl', 'rb') as f:
    S = pickle.load(f)
# all_data = np.concatenate((S.db.data, S.sim_sizes[:, None], S.todo[:, None], S.g.grid_pt_idx[:, None], S.g.null_truth), axis=1)
# np.save('2090.npy', all_data)
```

```python
df = pd.DataFrame(S.db.data)
```

```python
rename_dict = dict()
for k, v in S.db.slices.items():
    if isinstance(v, slice):
        for i in range(v.start, v.stop, 1 if v.step is None else v.step):
            rename_dict[i] = k + '_' + str(i)
    else:
        rename_dict[v] = k
df.rename(columns=rename_dict, inplace=True)
```

```python
df['sim_sizes'] = S.sim_sizes
df['todo'] = S.todo
df['grid_pt_idx'] = S.g.grid_pt_idx
for d in range(S.g.d):
    df[f'theta_{d}'] = S.g.thetas[S.g.grid_pt_idx, d]
    df[f'radii_{d}'] = S.g.radii[S.g.grid_pt_idx, d]

for i in range(S.g.null_truth.shape[1]):
    df[f'null_truth_{i}'] = S.g.null_truth[:, i]
```

```python
df.head()
```

```python
%%time
RR = df['grid_pt_idx'].sample(frac=0.1)
```

```python
%%time
RR.shape, df['grid_pt_idx'].isin(RR).sum()
```

```python
df.columns
```

```python
all_data = np.load('2090.npy')
```

## SQLite

```python

con = sqlite3.connect('tutorial.db')
```

```python
con.execute("DROP TABLE tiles")
con.execute("CREATE TABLE tiles(a REAL, b REAL, c REAL)")
con.execute("CREATE INDEX 'tiles_ordering' ON tiles(b)")
con.commit()
```

```python
rows = all_data[:int(1e6),:3]
```

```python
%%time
con.executemany("INSERT INTO tiles VALUES (?, ?, ?)", rows)
con.execute('select count(*) from tiles').fetchall()
con.commit()
```

```python
%%time
np.array(con.execute('select * from tiles order by b limit 1000000').fetchall())
```

```python
all_data[]
```

## duckdb

```python
import duckdb
import pyarrow as pa
import pandas as pd
con = duckdb.connect(database=':memory:')
# con.execute("DROP TABLE tiles")
con.execute("CREATE TABLE tiles(a REAL, b REAL, c REAL)")
con.execute("CREATE INDEX tiles_ordering ON tiles(b)")
con.commit()
```

```python
%%time
tbl = pa.Table.from_pandas(pd.DataFrame(rows))
con.execute('insert into tiles select * from tbl').fetchall()
```

```python
%%time
con.execute('select * from tiles order by b limit 1000000').fetchnumpy()
```

## redis??

```python
import redis
```

```python
r = redis.Redis(host='localhost', port=6379, db=0)
```

```python
r.set('foo', 'bar')
```

```python
r.get('foo')
```


