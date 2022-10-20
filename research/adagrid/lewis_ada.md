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

nb_util.setup_nb(pretty=True)
```

```python
import time
import jax
import os
import re
import pickle
import numpy as np
import jax.numpy as jnp
import scipy.spatial
import matplotlib.pyplot as plt
from confirm.mini_imprint import grid
from confirm.lewislib import grid as lewgrid
from confirm.lewislib import lewis, batch
from confirm.mini_imprint import binomial

import confirm.mini_imprint.lewis_drivers as lts

from rich import print as rprint
```

```python
# Configuration used during simulation
name = "4d_05"
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

```

```python
# Configuration used during simulation
# name = "3d_smaller2"
# params = {
#     "n_arms": 3,
#     "n_stage_1": 50,
#     "n_stage_2": 100,
#     "n_stage_1_interims": 2,
#     "n_stage_1_add_per_interim": 100,
#     "n_stage_2_add_per_interim": 100,
#     "stage_1_futility_threshold": 0.15,
#     "stage_1_efficacy_threshold": 0.7,
#     "stage_2_futility_threshold": 0.2,
#     "stage_2_efficacy_threshold": 0.95,
#     "inter_stage_futility_threshold": 0.6,
#     "posterior_difference_threshold": 0,
#     "rejection_threshold": 0.05,
#     "key": jax.random.PRNGKey(0),
#     "n_table_pts": 20,
#     "n_pr_sims": 100,
#     "n_sig2_sims": 20,
#     "batch_size": int(2**12),
#     "cache_tables": f"./{name}/lei_cache.pkl",
# }
```

```python
lei_obj = lewis.Lewis45(**params)
n_arm_samples = int(lei_obj.unifs_shape()[0])
```

```python
n_arms = params["n_arms"]
ns = np.concatenate(
    [np.ones(n_arms - 1)[:, None], -np.eye(n_arms - 1)],
    axis=-1,
)
null_hypos = [grid.HyperPlane(n, 0) for n in ns]
symmetry = []
for i in range(n_arms - 2):
    n = np.zeros(n_arms)
    n[i + 1] = 1
    n[i + 2] = -1
    symmetry.append(grid.HyperPlane(n, 0))

theta_min = -0.50
theta_max = 0.50
init_grid_size = 8
theta, radii = grid.cartesian_gridpts(
    np.full(n_arms, theta_min),
    np.full(n_arms, theta_max),
    np.full(n_arms, init_grid_size),
)
g_raw = grid.build_grid(theta, radii)
```

```python
target_grid_cost = 0.005
target_sim_cost = 0.005
target_alpha = 0.025
holderq = 6

grid_batch_size = 2**6 if jax.devices()[0].device_kind == "cpu" else 2**10
init_nsims = 1000
max_sim_double = 8
max_sim_size = init_nsims * 2 ** max_sim_double
seed = 0
src_key = jax.random.PRNGKey(seed)
key1, key2, key3 = jax.random.split(src_key, 3)

unifs = jax.random.uniform(key=key1, shape=(max_sim_size,) + lei_obj.unifs_shape())
unifs_order = np.arange(0, unifs.shape[1])
nB_global = 10
nB_tile = 20
bootstrap_idxs = {
    K: jnp.concatenate((
        jnp.arange(K)[None, :],
        jax.random.choice(key2, np.arange(K), shape=(nB_global, K), replace=True),
        jax.random.choice(key3, np.arange(K), shape=(nB_tile, K), replace=True)
    ))
    for K in (init_nsims * 2 ** np.arange(0, max_sim_double + 1))
}

batched_tune = lts.grouped_by_sim_size(lei_obj, lts.tunev, grid_batch_size)
batched_rej = lts.grouped_by_sim_size(lei_obj, lts.rejv, grid_batch_size)
batched_invert_bound = batch.batch(
    lambda *args: (binomial.invert_bound(*args),),
    grid_batch_size,
    in_axes=(None, 0, 0, None, None),
)
batched_many_rej = lts.grouped_by_sim_size(lei_obj, lts.rejvv, grid_batch_size)
```

```python
load_iter = 'latest'
if load_iter == 'latest':
    # find the file with the largest checkpoint index: name/###.pkl 
    available_iters = [int(fn[:-4]) for fn in os.listdir(name) if re.match(r'[0-9]+.pkl', fn)]
    load_iter = 0 if len(available_iters) == 0 else max(available_iters)

if load_iter == 0:
    g = grid.build_grid(
        theta, radii, null_hypos=null_hypos, symmetry_planes=symmetry, should_prune=True
    )
    sim_sizes = np.full(g.n_tiles, init_nsims)
    bootstrap_cvs = np.empty((g.n_tiles, 2 + nB_global), dtype=float)
    pointwise_target_alpha = np.empty(g.n_tiles, dtype=float)
    todo = np.ones(g.n_tiles, dtype=bool)
    # TODO: remove
    typeI_sum = None
    hob_upper = None
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
    todo = np.zeros(g.n_tiles, dtype=bool)
    todo[-1] = True
    # keep = np.ones(g.n_tiles, dtype=bool)
    # for d in range(3):
    #     keep &= (g.theta_tiles[:, d] > -1) & (g.theta_tiles[:, d] < 1)
    # g = grid.index_grid(g, keep)
    # pointwise_target_alpha = pointwise_target_alpha[keep]
    # sim_sizes = sim_sizes[keep]
    # bootstrap_cvs = bootstrap_cvs[keep]
    # typeI_sum = typeI_sum[keep] if typeI_sum is not None else None
    # hob_upper = hob_upper[keep] if hob_upper is not None else None
```

## The big run

```python
ada_step_size = 10 * grid_batch_size
ada_min_step_size = grid_batch_size
iter_max = 1000
cost_per_sim = 500e-9
for II in range(load_iter + 1, iter_max):
    if np.sum(todo) == 0:
        break

    print(f"starting iteration {II} with {np.sum(todo)} tiles to process")
    if cost_per_sim is not None:
        predicted_time = np.sum(sim_sizes[todo] * cost_per_sim)
        print(f"runtime prediction: {predicted_time:.2f} seconds")
        
    ########################################
    # Simulate any new or updated tiles. 
    ########################################

    start = time.time()
    pointwise_target_alpha[todo] = batched_invert_bound(
        target_alpha, g.theta_tiles[todo], g.vertices[todo], n_arm_samples, holderq
    )[0]
    print("inverting the bound took", time.time() - start)
    start = time.time()

    bootstrap_cvs_todo = lts.bootstrap_tune_runner(
        lei_obj,
        sim_sizes[todo],
        pointwise_target_alpha[todo],
        g.theta_tiles[todo],
        g.null_truth[todo],
        unifs,
        bootstrap_idxs,
        unifs_order,
    )
    bootstrap_cvs[todo, 0] = bootstrap_cvs_todo[:, 0]
    bootstrap_cvs[todo, 1:-1] = bootstrap_cvs_todo[:, 1:1+nB_global]
    bootstrap_cvs[todo, -1] = bootstrap_cvs_todo[:, 1+nB_global:].min(axis=1)
    worst_tile = np.argmin(bootstrap_cvs[:, 0])
    overall_cv = bootstrap_cvs[worst_tile, 0]
    cost_per_sim = (time.time() - start) / np.sum(sim_sizes[todo])
    todo[:] = False
    print("tuning took", time.time() - start)
    

    ########################################
    # Checkpoint 
    ########################################

    start = time.time()
    savedata = [g, sim_sizes, bootstrap_cvs, None, None, pointwise_target_alpha]
    if II % 10 == 0 or II <= load_iter + 5:
        with open(f"{name}/{II}.pkl", "wb") as f:
            pickle.dump(savedata, f)
    print("checkpointing took", time.time() - start)
    

    ########################################
    # Criterion step 1: is tuning impossible? 
    ########################################
    # try to estimate the number of refinements steps required to get to the
    # target alpha. for now, it's okay to slightly preference refinement over
    # adding sims because refinment gives more information in a sense.
    start = time.time()
    cost_to_refine = 2**n_arms
    sims_required_to_rej_once = 2 / pointwise_target_alpha - 1
    cost_to_rej_once = sims_required_to_rej_once / sim_sizes

    # if a tile always stops early, it's probably not interesting and we should
    # lean towards simulating more rather than more expensive refinement
    always_stops_early = bootstrap_cvs[:, 0] >= 1
    prefer_simulation = (cost_to_refine > cost_to_rej_once) & (always_stops_early)

    alpha_to_rej_once = 2 / (sim_sizes + 1)
    impossible = pointwise_target_alpha < alpha_to_rej_once
    impossible_refine = (impossible & (~prefer_simulation)) | (pointwise_target_alpha == 0)
    impossible_sim = impossible & prefer_simulation


    ########################################
    # Criterion step 2: what is the bias?
    ########################################
    bootstrap_min_cvs = np.min(bootstrap_cvs[:, :-1], axis=0)
    cv_std = bootstrap_min_cvs.std()
    worst_idxs = np.arange(worst_tile, worst_tile + 1)
    worst_many_rej = lts.grouped_by_sim_size(lei_obj, lts.rejvv, 1)
    worst_typeI_sum = worst_many_rej(
        sim_sizes[worst_idxs],
        (
            bootstrap_min_cvs[None, :],
            g.theta_tiles[worst_idxs],
            g.null_truth[worst_idxs],
        ),
        (unifs,),
        unifs_order,
    )
    bias = (worst_typeI_sum[0,0] - worst_typeI_sum[0,1:].mean()) / sim_sizes[worst_tile]
    
    ########################################
    # Criterion step 3: Refine tiles that are too large, deepen tiles that
    # cause too much bias.
    ########################################
    which_moresims = np.zeros(g.n_tiles, dtype=bool)
    which_refine = np.zeros(g.n_tiles, dtype=bool)
    tilewise_bootstrap_min_cv = bootstrap_cvs[:, -1]
    hob_theory_cost = target_alpha - pointwise_target_alpha
    
    if hob_theory_cost[worst_tile] > target_grid_cost or bias > target_sim_cost:
        sorted_orig_cvs = np.argsort(bootstrap_cvs[:, 0])
        sorted_bootstrap_cvs = np.argsort(tilewise_bootstrap_min_cv)
        dangerous_bootstrap = sorted_bootstrap_cvs[:ada_step_size]
        dangerous_cv = sorted_orig_cvs[:ada_min_step_size]

        which_refine[dangerous_bootstrap] = (hob_theory_cost[dangerous_bootstrap] > target_grid_cost)
        which_refine[dangerous_cv] = (hob_theory_cost[dangerous_cv] > target_grid_cost)
        which_refine |= impossible_refine

        which_moresims[dangerous_bootstrap] = bias > target_sim_cost
        which_moresims |= impossible_sim
        which_moresims &= ~which_refine
        which_moresims &= (sim_sizes < max_sim_size)

    ########################################
    # Report current status 
    ########################################
    report = dict(
        II=II,
        overall_cv=f"{overall_cv:.4f}",
        cv_std=f"{cv_std:.4f}",
        grid_cost=f"{hob_theory_cost[worst_tile]:.4f}",
        bias=f"{bias:.4f}",
        n_tiles=g.n_tiles,
        n_refine=np.sum(which_refine),
        n_refine_impossible=np.sum(impossible_refine),
        n_moresims=np.sum(which_moresims),
        n_moresims_impossible=np.sum(impossible_sim),
        # moresims_dist=np.unique(sim_multiplier, return_counts=True)
    )
    rprint(report)
    print("analysis took", time.time() - start)
    start = time.time()

    ########################################
    # Refine! 
    ########################################

    if (np.sum(which_refine) > 0 or np.sum(which_moresims) > 0) and II != iter_max - 1:

        sim_sizes[which_moresims] = sim_sizes[which_moresims] * 2
        todo[which_moresims] = True

        refine_tile_idxs = np.where(which_refine)[0]
        refine_gridpt_idxs = g.grid_pt_idx[refine_tile_idxs]
        new_thetas, new_radii, unrefined_grid, keep_tile_idxs = grid.refine_grid(
            g, refine_gridpt_idxs
        )
        new_grid = grid.build_grid(
            new_thetas,
            new_radii,
            null_hypos=g.null_hypos,
            symmetry_planes=symmetry,
            should_prune=True,
        )

        old_g = g
        g = grid.concat_grids(unrefined_grid, new_grid)

        sim_sizes = np.concatenate(
            [sim_sizes[keep_tile_idxs], np.full(new_grid.n_tiles, init_nsims)]
        )
        todo = np.concatenate(
            [todo[keep_tile_idxs], np.ones(new_grid.n_tiles, dtype=bool)]
        )
        bootstrap_cvs = np.concatenate(
            [
                bootstrap_cvs[keep_tile_idxs],
                np.zeros((new_grid.n_tiles, 2 + nB_global), dtype=float),
            ],
            axis=0,
        )
        pointwise_target_alpha = np.concatenate(
            [
                pointwise_target_alpha[keep_tile_idxs],
                np.empty(new_grid.n_tiles, dtype=float),
            ]
        )
        print("refinement took", time.time() - start)
        continue
    print("done!")
    savedata = [g, sim_sizes, bootstrap_cvs, None, None, pointwise_target_alpha]
    with open(f"{name}/{II}.pkl", "wb") as f:
        pickle.dump(savedata, f)
    break

```

```python
savedata = [g, sim_sizes, bootstrap_cvs, None, None, pointwise_target_alpha]
with open(f"{name}/{II}.pkl", "wb") as f:
    pickle.dump(savedata, f)
```

```python
dangerous = np.any(bootstrap_cvs[:, 1:] < overall_cv, axis=-1)
```

```python

```

```python

```

```python
bootstrap_min_cvs = np.min(bootstrap_cvs[:, 1:], axis=0)
cv_std = bootstrap_min_cvs.std()
bootstrap_cv_rej = batched_many_rej(
    sim_sizes[close_to_worst],
    (np.tile(bootstrap_min_cvs[None, :], (np.sum(close_to_worst), 1)),
    g.theta_tiles[close_to_worst],
    g.null_truth[close_to_worst],),
    (unifs,),
    unifs_order
)
```

```python
np.max()
```

```python

```

```python
close_to_worst.shape, g.n_tiles, sim_sizes.shape, g.theta_tiles.shape, g.null_truth.shape, np.tile(bootstrap_min_cvs[None, :], (close_to_worst.shape[0], 1)).shape
```

```python

```

```python
np.tile(bootstrap_cvs[None, :], (close_to_worst.shape[0], 1)).shape
```

```python
np.min(bootstrap_cvs, axis=0)
```

```python
bootstrap_cvs.shape
```

```python
typeI_sum = batched_rej(
    sim_sizes,
    (np.full(sim_sizes.shape[0], overall_cv),
    g.theta_tiles,
    g.null_truth,),
    unifs,
    unifs_order,
)

savedata = [
    g,
    sim_sizes,
    bootstrap_cvs,
    typeI_sum,
    hob_upper,
    pointwise_target_alpha
]
with open(f"{name}/final.pkl", "wb") as f:
    pickle.dump(savedata, f)
```

```python
typeI_est, typeI_CI = binomial.zero_order_bound(
    typeI_sum, sim_sizes, delta_validate, 1.0
)
typeI_bound = typeI_est + typeI_CI

hob_upper = binomial.holder_odi_bound(
    typeI_bound, g.theta_tiles, g.vertices, n_arm_samples, holderq
)
sim_cost = typeI_CI
hob_empirical_cost = hob_upper - typeI_bound
```

```python
worst_idx = np.argmax(typeI_est)
worst_tile = g.theta_tiles[worst_idx]
typeI_est[worst_idx], worst_tile
```

```python
worst_cv_idx = np.argmin(sim_cvs)
typeI_est[worst_cv_idx], sim_cvs[worst_cv_idx], g.theta_tiles[worst_cv_idx], pointwise_target_alpha[worst_cv_idx]
```

```python
plt.hist(typeI_est, bins=np.linspace(0.02,0.025, 100))
plt.show()
```

```python
np.sum((sim_cvs <= adafrac * overall_cv) & (hob_theory_cost > target_grid_cost))
```

```python
idxs = [worst_idx]
batched_sim(
    sim_sizes[idxs],
    np.full(1, sim_cvs[idxs]),
    g.theta_tiles[idxs],
    g.null_truth[idxs],
    unifs,
    unifs_order,
) / sim_sizes[idxs]
```

```python
# def pandemonium(field):
field = typeI_est
# for unplot_set in [{0, 1}, {1, 2}]:
for unplot_set in [{0}, {1}]:
    plot = list(set(range(n_arms)) - unplot_set)
    unplot = list(unplot_set)
    select = np.where(np.all(np.abs(g.theta_tiles[:, unplot] - worst_tile[unplot]) < 0.5, axis=-1))[
        0
    ]

    ordered_select = select[np.argsort(field[select])]
    print(ordered_select.shape[0])

    plt.figure(figsize=(6, 6))
    plt.title(r"$\hat{f}(\lambda^{*})$")
    plt.scatter(
        g.theta_tiles[ordered_select, plot[0]],
        g.theta_tiles[ordered_select, plot[1]],
        c=field[ordered_select],
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
for unplot in [0, 2]:
    plot = list({0, 1, 2} - {unplot})
    select = np.where(np.abs(g.theta_tiles[:, unplot] - worst_tile[unplot]) < 0.08)[0]

    ordered_select = select[np.argsort(sim_cvs[select])[::-1]]
    print(ordered_select.shape[0])

    plt.figure(figsize=(6, 6))
    plt.title(r"$\hat{f}(\lambda^{*})$")
    plt.scatter(
        g.theta_tiles[ordered_select, plot[0]],
        g.theta_tiles[ordered_select, plot[1]],
        c=sim_cvs[ordered_select],
        vmin=overall_cv,
        vmax=overall_cv * 2,
        s=20,
    )
    plt.colorbar()
    plt.xlabel(f"$\\theta_{plot[0]}$")
    plt.ylabel(f"$\\theta_{plot[1]}$")
    plt.show()
```

```python
(2.0 / g.radii.min()) ** 3 / g.n_tiles
```

## Helpful snippets

```python
checksims = np.where(sim_cvs <= adafrac * overall_cv)[0][:(10*grid_batch_size)]
typeI_sum_check = batched_rej(
    sim_sizes[checksims],
    (np.full(checksims.shape[0], overall_cv),
    g.theta_tiles[checksims],
    g.null_truth[checksims],),
    unifs,
    unifs_order,
)

np.min(typeI_sum_check / sim_sizes[checksims])

sim_sizes[checksims] = 128000
est, ci = binomial.zero_order_bound(pointwise_target_alpha[checksims] * sim_sizes[checksims], sim_sizes[checksims], 0.05, 1.0)
np.argmax(est + ci)
np.max(ci)
```

```python
plt.hist(np.sum(test_stat <= 0, axis=-1) / sim_sizes[idxs])
plt.show()
```

```python
overall_cv = np.min(sim_cvs)
rej1 = batched_rej(
    sim_sizes[idxs],
    (np.full(idxs.shape[0], overall_cv), g.theta_tiles[idxs], g.null_truth[idxs]),
    unifs,
    unifs_order,
)

key2 = jax.random.PRNGKey(seed + 1)
unifs2 = jax.random.uniform(key=key2, shape=(max_sim_size,) + lei_obj.unifs_shape())
rej2 = batched_rej(
    sim_sizes[idxs],
    (np.full(idxs.shape[0], overall_cv), g.theta_tiles[idxs], g.null_truth[idxs]),
    unifs2,
    unifs_order,
)

rej1, rej2

plt.hist((rej1 - rej2) / (sim_sizes[idxs]))
plt.show()
```

##### Bootstrap tuning

```python
batched_sim = lts.grouped_by_sim_size(lei_obj, lts.simv, grid_batch_size)

K = 1000

widx = np.argmin(sim_cvs)
idxs = np.arange(widx, widx + 1)
idxs = np.argsort(sim_cvs)[:10*grid_batch_size]
test_stats, best_arms = batched_sim(sim_sizes[idxs], (g.theta_tiles[idxs],), (unifs,), unifs_order)
false_test_stats = jnp.where(g.null_truth[np.arange(idxs.shape[0])[:, None], best_arms - 1], test_stats, 3.0)
_tunev = jax.vmap(lts._tune, in_axes=(0, None, None))
sim_cv = _tunev(false_test_stats, K, pointwise_target_alpha[idxs])
```

```python
idxs = np.argsort(sim_cvs)[:grid_batch_size]
bootstrap_sim_cvs = bootstrap_tune_runner(
    sim_sizes[idxs],
    pointwise_target_alpha[idxs],
    g.theta_tiles[idxs],
    g.null_truth[idxs],
    unifs,
    bootstrap_idxs,
    unifs_order,
)
bootstrap_cvs = bootstrap_sim_cvs.min(axis=0)
overall_cv = bootstrap_cvs[0]
overall_cv, bootstrap_cvs
```

```python

bootstrap_cv_rej = batched_many_rej(
    sim_sizes[idxs],
    (np.tile(bootstrap_cvs[None, :], (idxs.shape[0], 1)),
    g.theta_tiles[idxs],
    g.null_truth[idxs],),
    (unifs[:K],),
    unifs_order
)
```

```python
plt.hist(np.sqrt(bootstrap_cv_rej.var(axis=1)) / K)
plt.show()
```

```python
n_bootstrap = 30
bootstrap_idxs = jax.random.choice(key, np.arange(K), shape=(n_bootstrap, K), replace=True)
bootstrap_sim_cv = np.array([_tunev(false_test_stats[:, bootstrap_idxs[i]], K, target_alpha) for i in range(n_bootstrap)])
bootstrap_cv = bootstrap_sim_cv.min(axis=1)

check_idxs = np.arange(widx, widx + 1)
check_idxs = idxs#[grid_batch_size]
# bootstrap_cv_rej = rejvv(
#     lei_obj,
#     np.tile(bootstrap_cv[None, :], (check_idxs.shape[0], 1)),
#     g.theta_tiles[check_idxs],
#     g.null_truth[check_idxs],
#     unifs[:K],
#     unifs_order
# )[0]
bootstrap_cv_rej = batched_many_rej(
    sim_sizes[check_idxs],
    (np.tile(bootstrap_cv[None, :], (check_idxs.shape[0], 1)),
    g.theta_tiles[check_idxs],
    g.null_truth[check_idxs],),
    (unifs[:K],),
    unifs_order
)
```

```python
plt.hist(np.sqrt(bootstrap_cv_rej.var(axis=1)) / K)
```

```python
# with open(f'checkpoint/6.pkl', 'rb') as f:
#     g, sim_sizes, sim_cvs, typeI_sum, hob_upper, pointwise_target_alpha = pickle.load(f)

# typeI_est, typeI_CI = binomial.zero_order_bound(
#     typeI_sum, sim_sizes, delta_validate, 1.0
# )
# typeI_bound = typeI_est + typeI_CI
# hob_upper = binomial.holder_odi_bound(
#     typeI_bound, g.theta_tiles, g.vertices, n_arm_samples, holderq
# )


# sim_cost = typeI_CI
# hob_theory_cost = target_alpha - pointwise_target_alpha
# hob_empirical_cost = hob_upper - typeI_bound

# worst_tile = np.argmin(sim_cvs)
# which_refine = (
#     hob_theory_cost > max(adafrac * hob_theory_cost[worst_tile], target_grid_cost)
# ) & (
#     (hob_upper > adafrac * hob_upper[worst_tile]) | (sim_cvs == sim_cvs[worst_tile])
# )
# which_more_sims = (
#     typeI_CI > max(adafrac * typeI_CI[worst_tile], target_sim_cost)
# ) & (
#     (typeI_bound > adafrac * hob_upper[worst_tile])
#     | (sim_cvs == sim_cvs[worst_tile])
# )
# 

    # typeI_sum[todo] = batched_sim(
    #     sim_sizes[todo],
    #     np.full(todo.sum(), overall_cv),
    #     g.theta_tiles[todo],
    #     g.null_truth[todo],
    #     unifs,
    #     unifs_order,
    # )
    # import pickle

    # typeI_est, typeI_CI = binomial.zero_order_bound(
    #     typeI_sum, sim_sizes, delta_validate, 1.0
    # )
    # typeI_bound = typeI_est + typeI_CI
    # hob_upper = binomial.holder_odi_bound(
    #     typeI_bound, g.theta_tiles, g.vertices, n_arm_samples, holderq
    # )
    # sim_cost = typeI_CI
    # hob_empirical_cost = hob_upper - typeI_bound
    # which_refine = (
    #     hob_theory_cost > max(adafrac * hob_theory_cost[worst_tile], target_grid_cost)
    # ) & (
    #     (hob_upper > adafrac * hob_upper[worst_tile]) | (sim_cvs == sim_cvs[worst_tile])
    # )
    # which_more_sims = (
    #     typeI_CI > max(adafrac * typeI_CI[worst_tile], target_sim_cost)
    # ) & (
    #     (typeI_bound > adafrac * hob_upper[worst_tile])
    #     | (sim_cvs == sim_cvs[worst_tile])
    # )
        #
        # n_more_sims=np.sum(which_more_sims),
        # sim_cost=f"{sim_cost[worst_tile]:.4f}",
        #
        # sim_sizes[which_more_sims] *= 2
        # todo[which_more_sims] = True
        # typeI_sum = np.concatenate(
        #     [typeI_sum[keep_tile_idxs], np.zeros(new_grid.n_tiles, dtype=float)]
        # )
        # hob_upper = np.concatenate(
        #     [hob_upper[keep_tile_idxs], np.empty(new_grid.n_tiles, dtype=float)]
        # )
```

```python

    # close_to_worst = np.zeros(g.n_tiles, dtype=bool)
    # close_to_worst[np.argsort(bootstrap_cvs[:, 0])[:ada_step_size]] = True
    # close_to_worst[impossible] = False

    # hob_theory_cost = target_alpha - pointwise_target_alpha
    # which_refine = close_to_worst & (hob_theory_cost > target_grid_cost)
    # which_refine |= impossible_refine

    # concat_cvs = np.concatenate(
    #     (
    #         bootstrap_cvs[close_to_worst],
    #         np.tile(bootstrap_min_cvs[None, :], (np.sum(close_to_worst), 1)),
    #     ),
    #     axis=-1,
    # )
    # concat_typeI_sum = batched_many_rej(
    #     sim_sizes[close_to_worst],
    #     (
    #         concat_cvs,
    #         g.theta_tiles[close_to_worst],
    #         g.null_truth[close_to_worst],
    #     ),
    #     (unifs,),
    #     unifs_order,
    # )
    # tile_bootstrap_typeI_sum = concat_typeI_sum[:, :(n_bootstrap + 1)]
    # grid_bootstrap_typeI_sum = concat_typeI_sum[:, (n_bootstrap + 1):]

    # typeI_bias = np.zeros(g.n_tiles)
    # typeI_bias[close_to_worst] = (
    #     grid_bootstrap_typeI_sum[:, 0] - grid_bootstrap_typeI_sum[:, 1:].mean(axis=1)
    # ) / sim_sizes[close_to_worst]

    # typeI_tile_var = np.zeros(g.n_tiles)
    # typeI_tile_var[close_to_worst] = np.std(tile_bootstrap_typeI_sum, axis=-1) / sim_sizes[close_to_worst]
    
```
