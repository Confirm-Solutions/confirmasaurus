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

import re
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
# Run on CPU because a concurrent process is probably running on GPU.
jax.config.update('jax_platform_name', 'cpu')

import confirm.mini_imprint.lewis_drivers as lts
from confirm.lewislib import lewis
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
grid_batch_size = 2**6 if jax.devices()[0].device_kind == "cpu" else 2**10
batched_many_rej = lts.grouped_by_sim_size(lei_obj, lts.rejvv, grid_batch_size)
```

```python
n_arm_samples = int(lei_obj.unifs_shape()[0])
```

```python
import pickle
load_iter = 'latest'
# load_iter = 173
if load_iter == 'latest':
    # find the file with the largest checkpoint index: name/###.pkl 
    available_iters = [int(fn[:-4]) for fn in os.listdir(name) if re.match(r'[0-9]+.pkl', fn)]
    load_iter = 0 if len(available_iters) == 0 else max(available_iters)

if load_iter == 0:
    print('fail')
else:
    fn = f"{name}/{load_iter}.pkl"
    print(f'loading checkpoint {fn}')
    with open(fn, "rb") as f:
        (
            g,
            sim_sizes,
            bootstrap_cvs,
            typeI_sum,
            hob_upper,
            pointwise_target_alpha,
        ) = pickle.load(f)
!du -hs {name}
!df -h /
```

```python
# Configuration used during simulation
# TODO: things that should be saved in the future!!
n_arms = params['n_arms']
target_alpha = 0.025
target_grid_cost = 0.005
init_nsims = 1000
max_sim_double = 8
max_sim_size = init_nsims * 2 ** max_sim_double
seed = 0
key = jax.random.PRNGKey(seed)
unifs = jax.random.uniform(key=key, shape=(max_sim_size,) + lei_obj.unifs_shape())
unifs_order = np.arange(0, unifs.shape[1])
```

```python
# for i in range(203, 231):
#     with open(f'3d_withsims/{i}.pkl', 'rb') as f:
#         (
#             g,
#             sim_sizes,
#             bootstrap_cvs,
#             typeI_sum,
#             hob_upper,
#             pointwise_target_alpha,
#         ) = pickle.load(f)
#     print(i)
#     overall_cv = np.min(bootstrap_cvs[:,0])
#     print('loose bounds', np.sum((bootstrap_cvs[:,0] < 0.08) & (0.025 - pointwise_target_alpha > 0.002)))
    
```

```python
overall_cv = np.min(bootstrap_cvs[:,0])
print('number of tiles near critical: ', np.sum((bootstrap_cvs[:,0] < overall_cv + 0.01)))
print('    and with loose bounds', np.sum((bootstrap_cvs[:,0] < overall_cv + 0.01) & (0.025 - pointwise_target_alpha > 0.002)))
print(f'pct of sim sizes > 2000: {100 * np.mean(sim_sizes > 2000):.1f}%')
print(f'pct of alpha tight: {100 * np.mean(target_alpha - pointwise_target_alpha < target_grid_cost):.1f}%')
print(f'sim size distribution: {np.unique(sim_sizes, return_counts=True)}')
```

```python
np.unique(sim_sizes, return_counts=True)
```

```python
HH = [bootstrap_cvs[sim_sizes == K, 0] for K in np.unique(sim_sizes)]
plt.hist(
    HH,
    stacked=True,
    bins=np.linspace(0.05, 0.15, 100),
    label=[f'K={K}' for K in np.unique(sim_sizes)]
)
plt.legend()
plt.xlabel('$\lambda^*$')
plt.ylabel('number of tiles')
plt.show()
```

```python
HH = [np.repeat(bootstrap_cvs[sim_sizes == K, 0], K//1000) for K in np.unique(sim_sizes)]
plt.hist(
    HH,
    stacked=True,
    bins=np.linspace(0.05, 0.15, 100),
    label=[f'K={K}' for K in np.unique(sim_sizes)]
)
plt.legend()
plt.xlabel('$\lambda^*$')
plt.ylabel('number of tiles')
plt.show()
```

```python

```

```python
bootstrap_cvs[sim_sizes == 256000,-2]
```

```python
pointwise_target_alpha[bootstrap_cvs[:,1:-2].argmin(axis=0)]
```

```python
bootstrap_cvs[bootstrap_cvs[:,1:-2].argmin(axis=0), -1]
```

```python

bootstrap_cvs[:,0] < 0.05
```

```python
abc = [HHv.shape[0] for HHv in HH]
np.stack(([2, 16, 32, 64, 128, 256], abc / np.sum(abc) * 100), axis=1)
```

```python
refinement_done = target_alpha - pointwise_target_alpha < target_grid_cost
plt.hist(
    [bootstrap_cvs[refinement_done,0 ], bootstrap_cvs[~refinement_done,0]],
    stacked=True,
    bins=np.linspace(0.05, 0.15, 100),
    label=['fully refined', 'not fully refined'],
)
plt.legend()
plt.xlabel('$\lambda^*$')
plt.ylabel('number of tiles')
plt.show()
```

```python
worst_tile_idx = np.argmin(bootstrap_cvs[:,0])
worst_tile = g.theta_tiles[worst_tile_idx]
# def pandemonium(field):
field = bootstrap_cvs[:,0]
# for unplot_set in [{0, 1}, {1, 2}]:
for unplot_set in [{0}, {1}]:
    plot = list(set(range(n_arms)) - unplot_set)
    unplot = list(unplot_set)
    axis_slice = np.all(np.abs(g.theta_tiles[:, unplot] - (-0.01)) < 0.03, axis=-1)
    select = np.where(axis_slice & (field < 0.15))[0]

    ordered_select = select[np.argsort(field[select])[::-1]]
    print(ordered_select.shape[0])

    plt.figure(figsize=(6, 6))
    plt.title(r"$\lambda^{*}$")
    plt.scatter(
        g.theta_tiles[ordered_select, plot[0]],
        g.theta_tiles[ordered_select, plot[1]],
        c=field[ordered_select],
        vmin=0.05,
        vmax=0.15,
        s=20,
    )
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.colorbar()
    plt.xlabel(f"$\\theta_{plot[0]}$")
    plt.ylabel(f"$\\theta_{plot[1]}$")
    plt.show()
```

```python
import scipy.spatial
tree = scipy.spatial.KDTree(g.theta_tiles)
```

## Type I Error

```python
t0 = worst_tile[0]
ng = 300
t1 = np.linspace(-1, 1, ng)
t2 = np.linspace(-1, 1, ng)
TG = np.stack((np.full((ng, ng), t0), *np.meshgrid(t1, t2, indexing='ij'), ), axis=-1)
TGF = TG.reshape(-1, 3)
nearby = tree.query(TGF, k=5)
```

```python
idxs = np.unique(nearby[1])
typeI_sum = np.zeros(g.n_tiles)
typeI_sum[idxs] = batched_many_rej(
    sim_sizes[idxs],
    (np.full((idxs.shape[0], 1), overall_cv),
    g.theta_tiles[idxs],
    g.null_truth[idxs],),
    (unifs,),
    unifs_order
)[:,0]
typeI_err = typeI_sum / sim_sizes
```

```python
x = g.theta_tiles[idxs, 1]
y = g.theta_tiles[idxs, 2]
z = 100 * typeI_err[idxs]
alt_hypo = (x > t0) & (y > t0)
z[alt_hypo] = np.nan
plt.title(f'Type I error \% $\quad(\\theta_0 = {t0:.2f})$')
plt.scatter(x, y, c=z, s=5, vmin=0, vmax=2.5)
plt.scatter(y, x, c=z, s=5, vmin=0, vmax=2.5)
plt.colorbar()
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xticks([-1, -0.5, -0, 0.5, 1])
plt.yticks([-1, -0.5, -0, 0.5, 1])
plt.savefig('lei_pts.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
x = TG[...,1]
y = TG[...,2]
flip = TG[..., 2] > TG[..., 1]
z = 100 * typeI_err[nearby[1][:,0]].reshape(ng, ng)
z[flip] = z.T[flip]
alt_hypo = (TG[..., 1] > t0) & (TG[..., 2] > t0)
z[alt_hypo] = np.nan
levels = np.linspace(0, 2.5, 6)
plt.title(f'Type I error \% $\quad(\\theta_0 = {t0:.2f})$')
cntf = plt.contourf(x, y, z, levels=levels, extend='both')
plt.contour(
    x,
    y,
    z,
    levels=levels,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    extend='both'
)
cbar = plt.colorbar(cntf)#, ticks=[0, 1, 2, 3, 4])
# cbar.ax.set_yticklabels(["1", "10", "$10^2$", "$10^3$", "$10^4$"])
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.xticks([-1, -0.5, -0, 0.5, 1])
plt.yticks([-1, -0.5, -0, 0.5, 1])
plt.savefig('leit1e.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
x = TG[...,1]
y = TG[...,2]
flip = TG[..., 2] > TG[..., 1]
z = 100 * typeI_err[nearby[1][:,0]].reshape(ng, ng)
z[flip] = z.T[flip]
alt_hypo = (TG[..., 1] > t0) & (TG[..., 2] > t0)
z[alt_hypo] = np.nan
```

```python
bound_components = np.array([
    z[~alt_hypo] / 100,
    z[~alt_hypo] / 100,
    z[~alt_hypo] / 100,
    z[~alt_hypo] / 100,
    z[~alt_hypo] / 100,
    z[~alt_hypo] / 100,
]).reshape((6, -1))
```

```python
bound_components.shape
```

```python
bound_components.shape
```

```python
np.savetxt(f'P.csv', TGF[~alt_hypo.flatten()][:,1:].T, fmt="%s", delimiter=",")
np.savetxt(f'B.csv', bound_components.T, fmt="%s", delimiter=",")
```

## Grid density

```python
g.theta_tiles[np.argmin(bootstrap_cvs[:,0])]
```

```python
t2 = -0.84
ng = 50
t0 = np.linspace(-1, 1, ng)
t1 = np.linspace(-1, 1, ng)
TG = np.stack((*np.meshgrid(t0, t1, indexing='ij'), np.full((ng, ng), t2)), axis=-1)
TGF = TG.reshape(-1, 3)
```

```python
t0 = -0.01
ng = 71
t1 = np.linspace(-1, 1, ng)
t2 = np.linspace(-1, 1, ng)
TG = np.stack((np.full((ng, ng), t0), *np.meshgrid(t1, t2, indexing='ij'), ), axis=-1)
TGF = TG.reshape(-1, 3)
```

```python
nearby = tree.query_ball_point(TGF, 0.05)
nearby_count = [len(n) for n in nearby]
```

```python
x = TG[...,1]
y = TG[...,2]
z = np.array(nearby_count).reshape(ng, ng)
z[z == 0] = z.T[z == 0]
z = np.log10(z)
levels = np.linspace(0, 4, 9)
plt.title('$\log_{10}$(number of nearby tiles)')
cntf = plt.contourf(x, y, z, levels=levels, extend="both")
plt.contour(
    x,
    y,
    z,
    levels=levels,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    extend="both",
)
cbar = plt.colorbar(cntf, ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(["1", "10", "$10^2$", "$10^3$", "$10^4$"])
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.show()
```

```python
ada_step_size = 1 * grid_batch_size
close_to_worst = np.zeros(g.n_tiles, dtype=bool)
# close_to_worst[np.random.choice(np.arange(g.n_tiles), size)] = True
# close_to_worst[np.argsort(bootstrap_cvs[:, 0] * (1 - (sim_sizes >= 16000)))[:ada_step_size]] = True
close_to_worst[np.where(sim_sizes == 256000)[0]] = True
bootstrap_min_cvs = np.min(bootstrap_cvs[:, 0:], axis=0)
cv_std = bootstrap_min_cvs.std()
bootstrap_typeI_sum = batched_many_rej(
    sim_sizes[close_to_worst],
    (np.tile(bootstrap_min_cvs[None, :], (np.sum(close_to_worst), 1)),
    g.theta_tiles[close_to_worst],
    g.null_truth[close_to_worst],),
    (unifs,),
    unifs_order
)
typeI_std = np.zeros(g.n_tiles)
typeI_std[close_to_worst] = (bootstrap_typeI_sum / sim_sizes[close_to_worst, None]).std(axis=1)
bootstrap_min_cvs
```

```python
bootstrap_cvs[close_to_worst]
```

```python
overall_cv
```

```python
np.where(sim_sizes == 128000)[0][:32]
```

```python
bias = (bootstrap_typeI_sum[:,0] - bootstrap_typeI_sum[:, 1:].mean(axis=1)) / sim_sizes[close_to_worst]
std = bootstrap_typeI_sum[:, 1:].std(axis=1) / sim_sizes[close_to_worst]
bias, std
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

```python

# import confirm.lewislib.batch as batch
# import confirm.mini_imprint.binomial as binomial
# batched_invert_bound = batch.batch_all_concat(
#     lambda *args: (binomial.invert_bound(*args),),
#     grid_batch_size,
#     in_axes=(None, 0, 0, None, None),
# )
# pta = batched_invert_bound(
#     target_alpha, g.theta_tiles[:1], g.vertices[:1], n_arm_samples, holderq
# )[0]
# pta
```
