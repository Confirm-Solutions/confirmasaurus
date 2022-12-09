```python
import confirm.outlaw.nb_util as nb_util

nb_util.setup_nb()

import confirm.imprint as ip
import numpy as np
import matplotlib.pyplot as plt
```

```python
db = ip.db.DuckDB.connect("lewis1dslice.db")
```

```python
df = db.get_all()
```

```python
def verify_adagrid(df):
    inactive_ids = df.loc[~df["active"], "id"]
    assert inactive_ids.unique().shape == inactive_ids.shape

    parents = df["parent_id"].unique()
    parents_that_dont_exist = np.setdiff1d(parents, inactive_ids)
    inactive_tiles_with_no_children = np.setdiff1d(inactive_ids, parents)
    assert parents_that_dont_exist.shape[0] == 1
    assert parents_that_dont_exist[0] == 0
    assert inactive_tiles_with_no_children.shape[0] == 0
```

```python
verify_adagrid(df)
```

```python
g = ip.Grid(df).active()
```

```python
plt.figure(figsize=(10, 10))
plt.scatter(g.df["theta0"], g.df["theta1"], c=g.df["tie_bound"], cmap="viridis", s=4)
plt.colorbar()
plt.show()
```

```python
plt.figure(figsize=(10, 10))
plt.scatter(g.df["theta0"], g.df["theta1"], c=g.df["K"], cmap="viridis", s=4)
plt.colorbar()
plt.show()
```

```python
plt.figure(figsize=(10, 10))
plt.scatter(
    g.df["theta0"],
    g.df["theta1"],
    c=g.df["tie_cp_bound"] - g.df["tie_est"],
    cmap="viridis",
    s=4,
)
plt.colorbar()
plt.show()
```

```python

```
