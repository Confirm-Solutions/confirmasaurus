---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.6 ('base')
    language: python
    name: python3
---

```python
import confirm.outlaw.nb_util as nb_util
nb_util.setup_nb()

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
# Run on CPU because a concurrent process is probably running on GPU.
jax.config.update('jax_platform_name', 'cpu')

import lewis_tune_sim as lts
from confirm.lewislib import lewis
```

```python
import pickle
with open('3d/68.pkl', 'rb') as f:
    data = pickle.load(f)
g, sim_sizes, sim_cvs, _, _, pointwise_target_alpha = data
```

```python
# Configuration used during simulation
name = "3d"
params = {
    "n_arms": 3,
    "n_stage_1": 50,
    "n_stage_2": 100,
    "n_stage_1_interims": 2,
    "n_stage_1_add_per_interim": 100,
    "n_stage_2_add_per_interim": 100,
    "stage_1_futility_threshold": 0.15,
    "stage_1_efficacy_threshold": 0.7,
    "stage_2_futility_threshold": 0.2,
    "stage_2_efficacy_threshold": 0.95,
    "inter_stage_futility_threshold": 0.6,
    "posterior_difference_threshold": 0,
    "rejection_threshold": 0.05,
    "key": jax.random.PRNGKey(0),
    "n_table_pts": 20,
    "n_pr_sims": 100,
    "n_sig2_sims": 20,
    "batch_size": int(2**12),
    "cache_tables": f"./{name}/lei_cache.pkl",
}
lei_obj = lewis.Lewis45(**params)
```

```python
plt.hist(pointwise_target_alpha[sim_cvs == 0], bins=100)
plt.show()
```

```python
key = jax.random.PRNGKey(0)
unifs = jax.random.uniform(key=key, shape=(np.max(sim_sizes),) + lei_obj.unifs_shape())
unifs_order = np.arange(0, unifs.shape[1])

batch_size = 2**4
idxs = np.where(sim_cvs == 0)[0][:batch_size]
overall_cv = 0
simv = jax.jit(
    jax.vmap(lts.sim, in_axes=(None, 0, None, None)), static_argnums=(0,)
)
tunev = jax.jit(jax.vmap(lts.tune, in_axes=(None, 0, 0, 0, None, None)), static_argnums=(0,))
rejv = jax.jit(
    jax.vmap(lts.rej, in_axes=(None, 0, 0, 0, None, None)), static_argnums=(0,)
)
batched_sim = lts.grouped_by_sim_size(lei_obj, simv, batch_size, n_out=2)
batched_rej = lts.grouped_by_sim_size(lei_obj, rejv, batch_size)
batched_tune = lts.grouped_by_sim_size(lei_obj, tunev, batch_size)
```

```python
test_stats, best_arms = lts.simv(lei_obj, g.theta_tiles[idxs], unifs[:1000], unifs_order[:1000])
```

```python
alpha = pointwise_target_alpha[idxs[0]]
cv_idx = jnp.maximum(
    jnp.floor((unifs.shape[0] + 1) * jnp.maximum(alpha, 0)).astype(int) - 1, 0
)
cv_idx
```

```python
sortedts = jnp.sort(test_stats[0])
sortedts
```

```python
unifs.shape
```

```python
test_stats, best_arm = batched_sim(sim_sizes[idxs], [g.theta_tiles[idxs]], [unifs, unifs_order])
```

```python
test_stats.shape
```

```python
typeI_sum = batched_sim(
    sim_sizes[idxs],
    np.full(idxs.shape[0], overall_cv),
    g.theta_tiles[idxs],
    g.null_truth[idxs],
    unifs,
    unifs_order,
)
```
