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
from confirm.mini_imprint import binomial, checkpoint

import confirm.mini_imprint.lewis_drivers as lts

from rich import print as rprint
```

```python
# Configuration used during simulation
name = "4d_full"
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

theta_min = -1.0
theta_max = 1.0
init_grid_size = 8
theta, radii = grid.cartesian_gridpts(
    np.full(n_arms, theta_min),
    np.full(n_arms, theta_max),
    np.full(n_arms, init_grid_size),
)
g_raw = grid.build_grid(theta, radii)
```

```python
target_grid_cost = 0.002
target_sim_cost = 0.002
target_alpha = 0.025
holderq = 6

grid_batch_size = 2**6 if jax.devices()[0].device_kind == "cpu" else 2**10
init_nsims = 1000
max_sim_double = 8
max_sim_size = init_nsims * 2 ** max_sim_double
seed = 0
src_key = jax.random.PRNGKey(seed)
key1, key2, key3 = jax.random.split(src_key, 3)

unifs = jax.random.uniform(key=key1, shape=(max_sim_size,) + lei_obj.unifs_shape(), dtype=jnp.float32)
unifs_order = jnp.arange(0, unifs.shape[1])
nB_global = 50
nB_tile = 50
bootstrap_idxs = {
    K: jnp.concatenate((
        jnp.arange(K)[None, :],
        jax.random.choice(key2, K, shape=(nB_global, K), replace=True),
        jax.random.choice(key3, K, shape=(nB_tile, K), replace=True)
    )).astype(jnp.int32)
    for K in (init_nsims * 2 ** np.arange(0, max_sim_double + 1))
}

# batched_tune = lts.grouped_by_sim_size(lei_obj, lts.tunev, grid_batch_size)
# batched_rej = lts.grouped_by_sim_size(lei_obj, lts.rejv, grid_batch_size)
# batched_many_rej = lts.grouped_by_sim_size(lei_obj, lts.rejvv, grid_batch_size)


import confirm.mini_imprint.bound.binomial as ehbound
bwd_solver = ehbound.BackwardQCPSolver(n=n_arm_samples)
def invert_bound(alpha, theta_0, vertices, n):
    v = vertices - theta_0
    # NOTE: OPTIMIZATION POTENTIAL: if we ever need faster EH bounds, then we
    # can only run the optimizer at a single corner. The bound is still valid
    # because we're just using a suboptimal q.

    q_opt = jax.vmap(bwd_solver.solve, in_axes=(None, 0, None))(
        theta_0, v, alpha
    )
    return jnp.min(jax.vmap(ehbound.q_holder_bound_bwd, in_axes=(0, None, None, 0, None))(
        q_opt, n, theta_0, v, alpha
    ))
batched_invert_bound = batch.batch(
    jax.jit(jax.vmap(invert_bound, in_axes=(None, 0, 0, None)), static_argnums=(0, 3)),
    5*grid_batch_size,
    in_axes=(None, 0, 0, None),
)
```

```python
load_iter = 'latest'
# load_iter = -1
if load_iter == 'latest':
    # find the file with the largest checkpoint index: name/###.pkl 
    available_iters = [int(fn[:-4]) for fn in os.listdir(name) if re.match(r'[0-9]+.pkl', fn)]
    load_iter = -1 if len(available_iters) == 0 else max(available_iters)

if load_iter == -1:
    g = grid.build_grid(
        theta, radii, null_hypos=null_hypos, symmetry_planes=symmetry, should_prune=True
    )
    sim_sizes = np.full(g.n_tiles, init_nsims)
    bootstrap_cvs = np.empty((g.n_tiles, 4 + nB_global), dtype=float)
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

```python
ada_step_size = 10 * grid_batch_size
ada_min_step_size = grid_batch_size
iter_max = 10000
cost_per_sim = np.inf
for II in range(load_iter + 1, iter_max):
    if np.sum(todo) == 0:
        break

    print(f"starting iteration {II} with {np.sum(todo)} tiles to process")
    if cost_per_sim is not None:
        predicted_time = np.sum(sim_sizes[todo] * cost_per_sim)
        print(f"runtime prediction: {predicted_time:.2f}")

    ########################################
    # Simulate any new or updated tiles.
    ########################################

    start = time.time()
    pointwise_target_alpha[todo] = batched_invert_bound(
        target_alpha, g.theta_tiles[todo], g.vertices(todo), n_arm_samples
    )
    print(f"inverting the bound took {time.time() - start:.2f}s")
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
    # TODO: this indexing has been a source of bugs. it would be nice to a
    # tile-wise database tool that can give names to different values while
    # smoothly handling the refinement, possibly also sparsity?
    bootstrap_cvs[todo, 0] = bootstrap_cvs_todo[:, 0]
    bootstrap_cvs[todo, 1:1 + nB_global] = bootstrap_cvs_todo[:, 1 : 1 + nB_global]
    bootstrap_cvs[todo, 1 + nB_global] = bootstrap_cvs_todo[:, 1 + nB_global :].min(axis=1)
    bootstrap_cvs[todo, 2 + nB_global] = bootstrap_cvs_todo[:, 1 + nB_global :].mean(axis=1)
    bootstrap_cvs[todo, 3 + nB_global] = bootstrap_cvs_todo[:, 1 + nB_global :].max(axis=1)
    worst_tile = np.argmin(bootstrap_cvs[:, 0])
    overall_cv = bootstrap_cvs[worst_tile, 0]
    cost_per_sim = (time.time() - start) / np.sum(sim_sizes[todo])
    todo[:] = False
    print(f"tuning took {time.time() - start:.2f}s")

    ########################################
    # Checkpoint
    ########################################

    start = time.time()
    savedata = [g, sim_sizes, bootstrap_cvs, None, None, pointwise_target_alpha]
    with open(f"{name}/{II}.pkl", "wb") as f:
        pickle.dump(savedata, f)
    for old_II in checkpoint.exponential_delete(II):
        fp = f"{name}/{old_II}.pkl"
        if os.path.exists(fp):
            os.remove(fp)
    print(f"checkpointing took {time.time() - start:.2f}s")

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
    impossible_refine = (impossible & (~prefer_simulation)) | (
        pointwise_target_alpha == 0
    )
    impossible_sim = impossible & prefer_simulation

    ########################################
    # Criterion step 2: what is the bias?
    ########################################
    bootstrap_min_cvs = np.min(bootstrap_cvs[:, :-3], axis=0)
    cv_std = bootstrap_min_cvs.std()
    worst_stats = lts.one_stat(
        lei_obj,
        g.theta_tiles[worst_tile],
        g.null_truth[worst_tile],
        sim_sizes[worst_tile],
        unifs,
        unifs_order
    )
    worst_typeI_sum = (worst_stats[None, :] < bootstrap_min_cvs[:, None]).sum(axis=1)
    bias = (worst_typeI_sum[0] - worst_typeI_sum[1:].mean()) / sim_sizes[
        worst_tile
    ]

    ########################################
    # Criterion step 3: Refine tiles that are too large, deepen tiles that
    # cause too much bias.
    ########################################
    which_deepen = np.zeros(g.n_tiles, dtype=bool)
    which_refine = np.zeros(g.n_tiles, dtype=bool)
    alpha_cost = target_alpha - pointwise_target_alpha
    twb_min_cv = bootstrap_cvs[:, -3]
    twb_mean_cv = bootstrap_cvs[:, -2]
    twb_max_cv = bootstrap_cvs[:, -1]

    if alpha_cost[worst_tile] > target_grid_cost or bias > target_sim_cost:
        sorted_orig_cvs = np.argsort(bootstrap_cvs[:, 0])
        dangerous_cv = sorted_orig_cvs[:ada_min_step_size]

        inflated_min_cv = twb_min_cv#twb_mean_cv + (twb_min_cv - twb_mean_cv) * 6
        sorted_bootstrap_idxs = np.argsort(inflated_min_cv)
        dangerous_bootstrap = sorted_bootstrap_idxs[:ada_step_size]
        dangerous = np.union1d(dangerous_cv, dangerous_bootstrap)

        d_should_refine = alpha_cost[dangerous] > target_grid_cost
        deepen_likely_to_work = twb_mean_cv[dangerous] > twb_max_cv[worst_tile]
        d_should_deepen = deepen_likely_to_work & (sim_sizes[dangerous] < max_sim_size)
        which_refine[dangerous] = d_should_refine & (~d_should_deepen)
        which_deepen[dangerous] = d_should_deepen | (bias > target_sim_cost)

        which_refine |= impossible_refine
        which_deepen |= impossible_sim
        which_deepen &= ~which_refine
        which_deepen &= sim_sizes < max_sim_size
```

```python
np.sum(tilewise_bootstrap_min_cv < 0.044)
```

```python
twb_min_cv
```

```python


    ########################################
    # Report current status
    ########################################
    report = dict(
        II=II,
        overall_cv=f"{overall_cv:.5f}",
        cv_std=f"{cv_std:.4f}",
        grid_cost=f"{alpha_cost[worst_tile]:.5f}",
        bias=f"{bias:.5f}",
        n_tiles=g.n_tiles,
        n_refine=np.sum(which_refine),
        n_refine_impossible=np.sum(impossible_refine),
        n_moresims=np.sum(which_deepen),
        n_moresims_impossible=np.sum(impossible_sim),
        # moresims_dist=np.unique(sim_multiplier, return_counts=True)
    )
    rprint(report)
    print(f"analysis took", time.time() - start)
    start = time.time()

    ########################################
    # Refine!
    ########################################

    if (np.sum(which_refine) > 0 or np.sum(which_deepen) > 0) and II != iter_max - 1:
        if np.sum(which_refine) == 0:
            sim_sizes[which_deepen] = sim_sizes[which_deepen] * 2
            todo[which_deepen] = True

        refine_tile_idxs = np.where(which_refine)[0]
        refine_gridpt_idxs = g.grid_pt_idx[refine_tile_idxs]
        new_thetas, new_radii, keep_tile_idxs = grid.refine_grid(g, refine_gridpt_idxs)
        new_grid = grid.build_grid(
            new_thetas,
            new_radii,
            null_hypos=g.null_hypos,
            symmetry_planes=symmetry,
            should_prune=True,
        )

        old_g = g
        # NOTE: It would be possible to avoid concatenating the grid every
        # iteration. For particularly large problems, that might be a large win
        # in runtime. But the additional complexity is undesirable at the
        # moment. 
        g = grid.concat_grids(grid.index_grid(old_g, keep_tile_idxs), new_grid)

        sim_sizes = np.concatenate(
            [sim_sizes[keep_tile_idxs], np.full(new_grid.n_tiles, init_nsims)]
        )
        todo = np.concatenate(
            [todo[keep_tile_idxs], np.ones(new_grid.n_tiles, dtype=bool)]
        )
        bootstrap_cvs = np.concatenate(
            [
                bootstrap_cvs[keep_tile_idxs],
                np.zeros((new_grid.n_tiles, 4 + nB_global), dtype=float),
            ],
            axis=0,
        )
        pointwise_target_alpha = np.concatenate(
            [
                pointwise_target_alpha[keep_tile_idxs],
                np.empty(new_grid.n_tiles, dtype=float),
            ]
        )
        print(f"refinement took {time.time() - start:.2f}s")
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

# Calculate actual type I errors?
typeI_est, typeI_CI = binomial.zero_order_bound(
    typeI_sum, sim_sizes, delta_validate, 1.0
)
typeI_bound = typeI_est + typeI_CI

hob_upper = binomial.holder_odi_bound(
    typeI_bound, g.theta_tiles, g.vertices, n_arm_samples, holderq
)
sim_cost = typeI_CI
hob_empirical_cost = hob_upper - typeI_bound
worst_idx = np.argmax(typeI_est)
worst_tile = g.theta_tiles[worst_idx]
typeI_est[worst_idx], worst_tile
worst_cv_idx = np.argmin(sim_cvs)
typeI_est[worst_cv_idx], sim_cvs[worst_cv_idx], g.theta_tiles[worst_cv_idx], pointwise_target_alpha[worst_cv_idx]
plt.hist(typeI_est, bins=np.linspace(0.02,0.025, 100))
plt.show()

theta_0 = np.array([-1.0, -1.0, -1.0])      # sim point
v = 0.1 * np.ones(theta_0.shape[0])     # displacement
f0 = 0.01                               # Type I Error at theta_0
fwd_solver = ehbound.ForwardQCPSolver(n=n_arm_samples)
q_opt = fwd_solver.solve(theta_0=theta_0, v=v, a=f0) # optimal q
ehbound.q_holder_bound_fwd(q_opt, n_arm_samples, theta_0, v, f0)
```
