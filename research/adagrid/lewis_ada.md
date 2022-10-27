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
g = grid.build_grid(
    theta, radii, null_hypos=null_hypos, symmetry_planes=symmetry, should_prune=True
)
```

```python
import adastate
from criterion import Criterion

lei_obj = lewis.Lewis45(**params)
n_arm_samples = int(lei_obj.unifs_shape()[0])
```

```python
P = adastate.AdaParams(
    init_K=2**10,
    n_K_double=8,
    alpha_target=0.025,
    grid_target=0.002,
    bias_target=0.002,
    nB_global=50,
    nB_tile=50,
    step_size=2**14,
    tuning_min=20
)
D = adastate.init_data(P, lei_obj, 0)

```

```python
load_iter = 'latest'
S, load_iter, fn = adastate.load(name, load_iter)
if S is None:
    print('initializing')
    S = adastate.init_state(P, g)
S.todo[0] = True
```

```python
R = adastate.AdaRunner(P, lei_obj)
iter_max = 10000
cost_per_sim = np.inf
for II in range(load_iter + 1, iter_max):
    if np.sum(S.todo) == 0:
        break

    print(f"starting iteration {II} with {np.sum(S.todo)} tiles to process")
    total_effort = np.sum(S.sim_sizes[S.todo])
    predicted_time = total_effort * cost_per_sim
    print(f"runtime prediction: {predicted_time:.2f}")

    start = time.time()
    R.step(P, S, D)
    cost_per_sim = (time.time() - start) / total_effort
    print(f"step took {time.time() - start:.2f}s")

    start = time.time()
    adastate.save(f"{name}/{II}.pkl", S)
    adastate.clean_checkpoints(name, II)
    print(f"checkpointing took {time.time() - start:.2f}s")

    start = time.time()
    cr = Criterion(lei_obj, P, S, D)
    print(f'criterion took {time.time() - start:.2f}s')
    rprint(cr.report)

    start = time.time()
    if (np.sum(cr.which_refine) > 0 or np.sum(cr.which_deepen) > 0) and II != iter_max - 1:
        S.sim_sizes[cr.which_deepen] = S.sim_sizes[cr.which_deepen] * 2
        S.todo[cr.which_deepen] = True

        S = S.refine(P, cr.which_refine, null_hypos, symmetry)
        print(f"refinement took {time.time() - start:.2f}s")
```

```python

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
