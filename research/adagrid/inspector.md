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
import confirm.outlaw.nb_util as nb_util
nb_util.setup_nb()

import pickle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
# Run on CPU because a concurrent process is probably running on GPU.
jax.config.update('jax_platform_name', 'cpu')

import confirm.mini_imprint.lewis_drivers as lts
from confirm.lewislib import lewis

import adastate
from criterion import Criterion
from diagnostics import lamstar_histogram
```

```python
name = '4d_full'
params = {
    "n_arms": 4,
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
with open(f"./{name}/data_params.pkl", "rb") as f:
    P, D = pickle.load(f)
load_iter = 'latest'
S, load_iter, fn = adastate.load(name, load_iter)
```

```python
cr = Criterion(lei_obj, P, S, D)
```

```python
S.sim_sizes[cr.dangerous]
```

```python
cr.alpha_cost[cr.dangerous]
```

```python
np.sum(cr.impossible_refine_orig)
```

```python
cr.twb_worst_tile_lam_max
```

```python
cr.twb_worst_tile_lam_min
```

```python
assert S.twb_max_lam[cr.twb_worst_tile] == np.min(S.twb_max_lam)
assert S.twb_min_lam[cr.twb_worst_tile] == np.min(S.twb_min_lam[cr.ties])
```

```python
S.alpha0[cr.dangerous]
```

```python
S.B_lam[cr.dangerous].min(axis=1)
```

```python
twb_B_lamss = S.twb_min_lam[cr.B_lamss_idx]
twb_B_lamss
```

```python
S.alpha0[cr.B_lamss_idx]
```

```python
cr.B_lamss
```

```python
def find_queue_position(lam):
    return np.sum(cr.inflated_min_lam[:, None] < lam[None, :], axis=0)
```

```python
S.twb_min_lam[np.argsort(S.orig_lam)[0]]
```

```python
up_next = np.argsort(S.orig_lam)[:10000]
S.alpha0[up_next].argmax(), S.sim_sizes[up_next].max()
```

```python
sorted_ordering = np.sort(cr.inflated_min_lam)
query = cr.inflated_min_lam[np.argsort(S.orig_lam)[:1000]]
overall_priority = jnp.searchsorted(sorted_ordering, query)
print("overall driver priority", overall_priority)

```

```python
np.maximum.accumulate(overall_priority)
```

```python
print('overall_lam', cr.overall_lam)
print('bias driver priority', find_queue_position(twb_B_lamss))
B_min = S.B_lam.min(axis=1)
bias_bad = B_min < cr.overall_lam
print('n bias bad', np.sum(bias_bad))
n_critical = np.sum((S.orig_lam < cr.overall_lam + 0.01))
n_loose = np.sum(
    (S.orig_lam < cr.overall_lam + 0.01)
    & (P.alpha_target - S.alpha0 > P.grid_target)
)
print(f"number of tiles near critical: {n_critical}")
print(f"    and with loose bounds {n_loose}")
# for i in range(10):
#     dangerous = np.sum(cr.inflated_min_lam[bias_bad] < cr.overall_lam)
#     collateral = np.sum(cr.inflated_min_lam < cr.overall_lam)
#     print(f'inflation factor {i}')
#     print(f'    dangerous tiles caught: {dangerous}')
#     print(f'    collateral tiles caught: {collateral}')

print('lambda**B', cr.B_lamss)
total_effort = np.sum(S.sim_sizes)
for K in np.unique(S.sim_sizes):
    sel = S.sim_sizes == K
    count = np.sum(sel)
    print(f"K={K}:")
    print(f'    count={count / 1e6:.3f}m')
    print(f'    lambda**B[K]={S.B_lam[sel].min(axis=0)}')
    print(f'    min lambda*B[K]={np.min(S.B_lam[sel].min(axis=1)):.4f}')
    print(f'    min lambda*b[K]={np.min(S.twb_min_lam):.4f}')
    effort = K * count / total_effort
    print(f'    % effort={100 * effort:.4f}') 
```

```python
S.db.data[cr.twb_worst_tile, -3:]
```

```python
S.alpha0[S.twb_min_lam < 0.04359697]
```

```python
S.twb_min_lam[np.argsort(S.orig_lam)[:1000]]
```

```python
plt.figure(figsize=(10, 10), constrained_layout=True)
plt.subplot(2,2, 1)
plt.title('$min(\lambda^*_B)$')
lamstar_histogram(S.B_lam.min(axis=1), S.sim_sizes)
for i, (field, title) in enumerate([(S.orig_lam, '$\lambda^{*}$'), (S.twb_min_lam, '$min(\lambda^*_b)$'), (S.twb_mean_lam, '$mean(\lambda^*_b)$')]):
    plt.subplot(2,2,i + 2)
    plt.title(title)
    lamstar_histogram(field, S.sim_sizes)
plt.show()
```

## Resimulation

```python
import pandas as pd
friends = np.where(bootstrap_cvs[:,0] < 0.045)[0]
print(pd.DataFrame(sim_sizes[friends]).describe())
print(pd.DataFrame(pointwise_target_alpha[friends]).describe())
```

```python
seed = 0
src_key = jax.random.PRNGKey(seed)
key1, key2, key3 = jax.random.split(src_key, 3)

unifs = jax.random.uniform(key=key1, shape=(adap.max_sim_size,) + lei_obj.unifs_shape(), dtype=jnp.float32)
unifs_order = jnp.arange(0, unifs.shape[1])
nB_global = 30
nB_tile = 40
bootstrap_idxs = {
    K: jnp.concatenate((
        jnp.arange(K)[None, :],
        jax.random.choice(key2, K, shape=(nB_global, K), replace=True),
        jax.random.choice(key3, K, shape=(nB_tile, K), replace=True)
    )).astype(jnp.int32)
    for K in (adap.init_K * 2 ** np.arange(0, adap.n_sim_double + 1))
}
```

```python
print('hi')
```

```python
which = friends[:4]
lamstar = lts.bootstrap_tune_runner(
    lei_obj,
    sim_sizes[which],
    pointwise_target_alpha[which],
    g.theta_tiles[which],
    g.null_truth[which],
    unifs,
    bootstrap_idxs,
    unifs_order,
    grid_batch_size=4
)
```

```python
stats = np.random.rand(3, 1000)
```

```python
from confirm.lewislib import batch
grid_batch_size=4
def printer(x, y, z):
    print(x.shape, y.shape, z.shape)
    return 0
tunev = jax.jit(jax.vmap(jax.vmap(lts.tune, in_axes=(None, 0, None)), in_axes=(0, None, 0)))
batched_tune = batch.batch(
    batch.batch(tunev, 10, in_axes=(None, 0, None), out_axes=(1,)),
    grid_batch_size, in_axes=(0, None, 0)
)
batched_tune(stats, bootstrap_idxs[1000], np.array([0.025, 0.025, 0.025])).shape
```

```python
bootstrap_idxs[1000].shape
```

```python
batch.batch(lts.tunev, 10, in_axes=(None, 0, None))(stats[0], bootstrap_idxs[1000], 0.025).shape
```

```python
tunev(stats, bootstrap_idxs[1000], np.full(3, 0.025)).shape
```

```python
bootstrap_idxs[1000].shape
```

## Look at the worst case from bootstrap group 1.

- $\lambda^*$ is the tile-wise threshold
- $\lambda^{**}$ is the global minimum threshold.
-

- TODO: what is the right notation for the different $\lambda$??

These are points that will drive down $\lambda^*_B$

```python
bootstrap_mins2 = bootstrap_cvs[:,1:-2].min(axis=1)
trixy = np.argsort(bootstrap_mins2)[:100]
print(bootstrap_mins2[trixy])
print(bootstrap_cvs[trixy, 0])
print(bootstrap_cvs[trixy, -2])
print(bootstrap_cvs[trixy, -1])
```

```python
trixy = bootstrap_cvs[:, :-2].argmin(axis=0)
print(bootstrap_mins2[trixy])
print(bootstrap_cvs[trixy, 0])
print(bootstrap_cvs[trixy, -2])
print(bootstrap_cvs[trixy, -1])
```

```python
pointwise_target_alpha[trixy], sim_sizes[trixy], g.radii[g.grid_pt_idx[trixy]]
```

```python
import scipy.spatial
tree = scipy.spatial.KDTree(g.theta_tiles)
```

```python
worst_tile_idx = np.argmin(bootstrap_cvs[:,0])
worst_tile = g.theta_tiles[worst_tile_idx]

slice_pt = worst_tile
plot_dims = [0, 1]
unplot_dims = list(set(range(g.d)) - set(plot_dims))

slicex = [-1, 1]
slicey = [-1, 1]
nx = ny = 100
xvs = np.linspace(*slicex, nx)
yvs = np.linspace(*slicey, ny)
grid = np.stack(np.meshgrid(xvs, yvs, indexing='ij'), axis=-1)
full_grid = np.empty((nx * ny, g.d))
full_grid[:, plot_dims] = grid.reshape(-1, 2)
full_grid[:, unplot_dims] = slice_pt[unplot_dims]
```

```python
closest_idx[1]
```

```python
closest_idx = tree.query(full_grid)
closest_idx
```

```python
eval_pts = 
```

```python
# worst_tile_idx = np.argmin(bootstrap_cvs[:,0])
# worst_tile = g.theta_tiles[worst_tile_idx]
# # def pandemonium(field):
# field = bootstrap_cvs[:,0]
# # for unplot_set in [{0, 1}, {1, 2}]:
# for unplot_set in [{0}, {1}]:
#     plot = list(set(range(n_arms)) - unplot_set)
#     unplot = list(unplot_set)
#     axis_slice = np.all(np.abs(g.theta_tiles[:, unplot] - (-0.01)) < 0.03, axis=-1)
#     select = np.where(axis_slice & (field < 0.15))[0]

#     ordered_select = select[np.argsort(field[select])[::-1]]
#     print(ordered_select.shape[0])

#     plt.figure(figsize=(6, 6))
#     plt.title(r"$\lambda^{*}$")
#     plt.scatter(
#         g.theta_tiles[ordered_select, plot[0]],
#         g.theta_tiles[ordered_select, plot[1]],
#         c=field[ordered_select],
#         vmin=0.05,
#         vmax=0.15,
#         s=20,
#     )
#     plt.xlim([-1, 1])
#     plt.ylim([-1, 1])
#     plt.colorbar()
#     plt.xlabel(f"$\\theta_{plot[0]}$")
#     plt.ylabel(f"$\\theta_{plot[1]}$")
#     plt.show()
```
