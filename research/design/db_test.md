---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
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

import adastate

with open("../adagrid/4d_full/3880.pkl", "rb") as f:
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
            rename_dict[i] = k + "_" + str(i - v.start)
    else:
        rename_dict[v] = k
df.rename(columns=rename_dict, inplace=True)

df["sim_sizes"] = S.sim_sizes
df["todo"] = S.todo
df["grid_pt_idx"] = S.g.grid_pt_idx
for d in range(S.g.d):
    df[f"theta_{d}"] = S.g.thetas[S.g.grid_pt_idx, d]
    df[f"radii_{d}"] = S.g.radii[S.g.grid_pt_idx, d]

for i in range(S.g.null_truth.shape[1]):
    df[f"null_truth_{i}"] = S.g.null_truth[:, i]
```

```python
df.head()
```

```python
chunk = df[:100000]
```

## SQLite

```python
con = sqlite3.connect("tutorial.db")
```

```python
%%time
chunk.to_sql("tiles", con, if_exists="replace")
```

```python
con.execute("select count(*) from tiles").fetchall()
```

```python
%%time
dfload = pd.read_sql("select * from tiles", con)
```

```python
dfload.head()
```

```python
%%time
con.execute(
    "select " + ",".join([f"min(B_lam_{i})" for i in range(50)]) + " from tiles"
).fetchall()
```

## duckdb

```python
chunk = df[:]
chunk.shape
```

```python
import duckdb
import pyarrow as pa
import pandas as pd
import pickle

con = duckdb.connect()
```

```python
%%time
# con.execute('drop table tiles')
in_tbl = pa.Table.from_pandas(chunk)
con.execute("create table tiles as select * from chunk")
# con.execute('create index twb_min_lam_idx on tiles (twb_min_lam)')
```

```python
%%time
out = con.execute("select * from tiles order by twb_min_lam limit 10000").fetch_df()
out.head()
```

```python
%%time
con.execute(
    "select " + ",".join([f"min(B_lam_{i})" for i in range(50)]) + " from tiles"
).fetchall()
```
