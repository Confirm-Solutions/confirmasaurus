---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.10.6 ('confirm')
    language: python
    name: python3
---

```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from confirm.mini_imprint import batch, grid, newlib, adagrid, driver
```

```python
@jax.jit
def _sim(samples, theta, null_truth):
    return jnp.where(
        null_truth[:, None, 0],
        # negate so that we can do a less than comparison
        -(theta[:, None, 0] + samples[None, :]),
        jnp.inf,
    )


class ZTest1D:
    def __init__(self, seed, max_K, sim_batch_size=2048):
        self.family = "normal"
        self.sim_batch_size = sim_batch_size
        self.dtype = jnp.float32

        # sample normals and then compute the CDF to transform into the
        # interval [0, 1]
        key = jax.random.PRNGKey(seed)
        self.samples = jax.random.normal(key, shape=(max_K,), dtype=self.dtype)
        self._sim_batch = batch.batch(
            _sim, self.sim_batch_size, in_axes=(0, None, None), out_axes=(1,)
        )

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return self._sim_batch(self.samples[begin_sim:end_sim], theta, null_truth)
```

## Calculate Type I Error

```python
K = 8192
model = ZTest1D(seed=0, max_K=K)

N = 100
theta, radii = grid.cartesian_gridpts([-1], [1], [N])
g = newlib.init_grid(theta, radii, K).add_null_hypo(0).prune()

# TODO: is there any problem from using the same seed with the bootstrap
# indices and the simulations?
dd = driver.Driver(model)
```

```python
# lam = -1.96 because we negated the statistics so we can do a less thanj
# comparison.
rej_df = dd.rej(g, -1.96)
```

```python
plt.plot(g.df["theta0"], rej_df["TI_est"], "b-o", markersize=2)
plt.plot(g.df["theta0"], rej_df["TI_cp_bound"], "k-o", markersize=2)
plt.plot(g.df["theta0"], rej_df["TI_bound"], "r-o", markersize=2)
plt.show()
```

## Adagrid Tuning

```python
init_K = 2048
n_K_double = 4
model = ZTest1D(seed=1, max_K=init_K * 2**n_K_double)


N = 10
theta, radii = grid.cartesian_gridpts([-1], [1], [N])
g = newlib.init_grid(theta, radii, init_K).add_null_hypo(0).prune()

nB = 6
tuning_min_idx = 20
ada = adagrid.AdagridDriver(model, init_K, n_K_double, nB, bootstrap_seed=2)
```

```python
df_tune = ada.bootstrap_tune(g)
```

```python
df_tune.iloc[df_tune["lams"].argmin()]
```

```python
lams_bias = (
    df_tune["lams"].min(axis=0)
    - df_tune[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0).mean()
)
lams_bias
```

```python
def f(x):
    return df_tune.iloc[x]
```

```python
dd.stats(g).apply()
```

```python
df_tune["impossible"] = df_tune["alpha0"] < (tuning_min_idx + 1) / (g.df["K"] + 1)
```

```python
df_tune[["twb_min_lams", "twb_mean_lams", "twb_max_lams", "lams"]]
```

```python
plt.plot(g.df["theta0"], df_tune["lams"], "k-o")
plt.show()
```

```python

```

```python
import matplotlib.pyplot as plt

stats = dd.stats(g).iloc[0]
plt.hist(stats[0])
plt.show()
```

```python

```
