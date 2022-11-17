```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from confirm.mini_imprint import batch, grid, adagrid, driver, db
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
Hs = [grid.HyperPlane(np.array([-1]), 0)]
g = grid.init_grid(theta, radii, K).add_null_hypos(Hs).prune()

# TODO: is there any problem from using the same seed with the bootstrap
# indices and the simulations?
dd = driver.Driver(model)

# lam = -1.96 because we negated the statistics so we can do a less thanj
# comparison.
rej_df = dd.rej(g.df, -1.96)

plt.plot(g.df["theta0"], rej_df["TI_sum"] / rej_df["K"], "b-o", markersize=2)
plt.plot(g.df["theta0"], rej_df["TI_cp_bound"], "k-o", markersize=2)
plt.plot(g.df["theta0"], rej_df["TI_bound"], "r-o", markersize=2)
plt.show()
```

## Adagrid Tuning

```python
init_K = 2048
n_K_double = 4
model = ZTest1D(seed=1, max_K=init_K * 2**n_K_double)


N = 1
theta, radii = grid.cartesian_gridpts([-1], [1], [N])
Hs = [grid.HyperPlane(np.array([-1]), 0)]
g = grid.init_grid(theta, radii, K).add_null_hypos(Hs).prune()

ada, reports = adagrid.adagrid(model, g, nB=5)
```

```python
evolution = pd.DataFrame(reports)
plt.plot(evolution["i"], evolution["bias"].astype(float))
plt.show()
plt.plot(evolution["i"], evolution["grid_cost"].astype(float))
plt.show()
plt.plot(evolution["i"], evolution["std_tie"].astype(float))
plt.show()
```

```python
all = ada.tiledb.get_all()
all = all.loc[all["active"]]
all.nsmallest(10, "orderer")
```

```python
import scipy.stats

lamss = all["lams"].min()
true_err = 1 - scipy.stats.norm.cdf(-lamss)
lamss, true_err
```

```python
plt.plot(all["theta0"], all["lams"], "ko")
plt.show()
plt.plot(all["theta0"], all["K"], "ko")
plt.show()
plt.plot(all["theta0"], all["alpha0"], "ko")
plt.show()
```
