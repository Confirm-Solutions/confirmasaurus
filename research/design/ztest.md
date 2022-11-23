```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

import confirm.imprint as ip
from confirm.models.ztest import ZTest1D
```

## Validation

```python
g = ip.cartesian_grid([-1], [1], n=[100], null_hypos=[ip.hypo("x < 0")])
# lam = -1.96 because we negated the statistics so we can do a less thanj
# comparison.
lam = -1.96
K = 8192
rej_df = ip.validate(ZTest1D, g, lam, K=K)
true_err = 1 - scipy.stats.norm.cdf(-g.get_theta()[:, 0] - lam)

plt.plot(g.df["theta0"], rej_df["tie_est"], "bo", markersize=2)
plt.plot(g.df["theta0"], rej_df["tie_cp_bound"], "ko", markersize=2)
plt.plot(g.df["theta0"], rej_df["tie_bound"], "ro", markersize=2)
plt.plot(g.df["theta0"], true_err, "r-o", markersize=2)
plt.show()
```

```python
std = scipy.stats.binom.std(n=K, p=true_err) / K
std
```

```python
err = np.abs(rej_df["tie_est"] - true_err).values
err
```

```python
err / std
```

```python
plt.plot(std)
plt.plot(err)
plt.show()
```

## Tuning

```python
tune_df = ip.tune(ZTest1D, g)
print("lambda**: ", tune_df["lams"].min())
plt.plot(g.df["theta0"], tune_df["lams"], "k-o", markersize=2)
plt.show()
```

## Adagrid Tuning

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
n_iter, reports, ada = ip.ada_tune(ZTest1D, g=g, nB=5)
```

```python
import scipy.stats

g = ip.Grid(ada.db.get_all())
ga = g.active()
lamss = ga.df["lams"].min()
true_err = 1 - scipy.stats.norm.cdf(-lamss)
lamss, true_err
```

```python
evolution = pd.DataFrame(reports)
# Figure plotting bias, grid_cost and std_tie
fig, ax = plt.subplots(3, 2, figsize=(8, 12), constrained_layout=True)
ax[0][0].plot(evolution["i"], evolution["bias_tie"], "o-")
ax[0][0].set_title("Bias")
ax[0][1].plot(evolution["i"], evolution["grid_cost"], "o-")
ax[0][1].set_title("Grid Cost")
ax[1][0].plot(evolution["i"], evolution["std_tie"], "o-")
ax[1][0].set_title("Std Tie")
ax[1][1].plot(ga.get_theta()[:, 0], ga.get_radii()[:, 0], "bo", markersize=3)
ax[1][1].set_title("Radius")
ax[2][0].plot(ga.get_theta()[:, 0], ga.df["K"], "bo", markersize=3)
ax[2][0].set_title("K")
ax[2][1].plot(ga.get_theta()[:, 0], ga.df["alpha0"], "bo", markersize=3)
ax[2][1].set_title("alpha0")
plt.show()
```
