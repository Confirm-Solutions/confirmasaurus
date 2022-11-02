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
import confirm.berrylib.util as util

util.setup_nb(pretty=False)

import time
from scipy.special import logit, expit
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import jax.numpy as jnp
import warnings
import confirm.berrylib.fast_inla as fast_inla
import confirm.mini_imprint.binomial as binomial
import confirm.mini_imprint.grid as grid
import confirm.mini_imprint.execute as execute

import jax
```

```python
def dots_plot(g, typeI_upper_bound, hob):
    plt.subplots(1, 2, figsize=(7, 3.0), constrained_layout=True)
    plt.subplot(1, 2, 1)
    plt.scatter(g.theta_tiles[:, 0], g.theta_tiles[:, 1], c=hob, s=10)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(g.theta_tiles[:, 0], g.theta_tiles[:, 1], c=typeI_upper_bound, s=10)
    plt.colorbar()
    plt.show()


def dots_plot2(g, typeI_upper_bound, hob):
    plt.scatter(g.theta_tiles[:, 0], g.theta_tiles[:, 1], c=hob, s=10)
    plt.colorbar()
```

```python
n_arms = 2
n_arm_samples = 35
fi = fast_inla.FastINLA(n_arms=n_arms, critical_value=0.99)
rejection_table = binomial.build_rejection_table(
    n_arms, n_arm_samples, fi.rejection_inference
)
accumulator = binomial.binomial_accumulator(
    lambda data: binomial.lookup_rejection(rejection_table, data[..., 0])
)
```

```python
n_theta_1d = 4
theta_min = -3.5
theta_max = 1.0

null_hypos = [
    grid.HyperPlane(-np.identity(n_arms)[i], -logit(0.1)) for i in range(n_arms)
]
theta, radii = grid.cartesian_gridpts(
    np.full(n_arms, theta_min), np.full(n_arms, theta_max), np.full(n_arms, n_theta_1d)
)
g_raw = grid.build_grid(theta, radii)
start_grid = grid.prune(grid.intersect_grid(g_raw, null_hypos))
```

```python
seed = 10

target_hob_cost = 0.001
target_hob_rel_bound = 0.3
target_sim_cost = 0.001
target_sim_rel_bound = 0.3
# N_max = int(2e5)

iter_max = 100
```

```python
from rich import print as rprint
```

```python
# plt.subplots(
#     iter_max // 3, 3, figsize=(10.0, 3.5 * iter_max / 3), constrained_layout=True
# )

A = execute.ada_setup(start_grid, n_initial_sims=5000, delta=0.01, holderq=6)
cur_bound = np.inf

report_history = []
for ada_i in range(iter_max):
    # np.random.seed(seed)
    start = time.time()
    old_total = A.total_sims
    A = execute.ada_simulate(A, accumulator, n_arm_samples)
    sim_runtime = time.time() - start
    assert np.all(A.sim_sizes == A.target_sim_sizes)

    # plt.subplot(iter_max // 3, 3, ada_i + 1)
    # dots_plot2(A.g, typeI_upper_bound, hob_upper)
    if ada_i == iter_max - 1:
        break

    typeI_upper_bound = A.typeI_est + A.typeI_CI
    cur_bound = np.max(A.hob_upper)
    worst_tile = np.argmax(A.hob_upper)
    should_refine = (
        A.hob_upper[worst_tile] - typeI_upper_bound[worst_tile]
        > typeI_upper_bound[worst_tile] - A.typeI_est[worst_tile]
    )

    hob_target_bound = np.max(typeI_upper_bound)
    hob_expensive = A.hob_upper > hob_target_bound + target_hob_cost
    hob_loose = (
        A.hob_upper - typeI_upper_bound
    ) / typeI_upper_bound > target_hob_rel_bound
    hob_tiny = A.hob_upper < 0.2 * hob_target_bound
    which_refine = hob_expensive | (hob_loose & (~hob_tiny))

    sim_target_bound = np.max(A.typeI_est)
    sim_expensive = typeI_upper_bound > sim_target_bound + target_sim_cost
    sim_loose = (typeI_upper_bound - A.typeI_est) / (
        A.typeI_est + 1e-9
    ) > target_sim_rel_bound
    sim_tiny = typeI_upper_bound < 0.2 * sim_target_bound
    more_sims = sim_expensive | (sim_loose & (~sim_tiny))

    report = dict(
        iter=ada_i,
        cur_bound=f"{cur_bound:.4f}",
        n_tiles=A.g.n_tiles,
        total_sims=A.total_sims,
        new_sims=f"{(A.total_sims - old_total) / 1000000:.1f}m",
        total_sims_so_far=f"{A.total_sims / 1000000:.1f}m",
    )

    # if (np.sum(which_refine) > 0) and (should_refine or np.sum(more_sims) == 0):
    A.target_sim_sizes[more_sims] *= 2
    report["n_add_sims"] = np.sum(more_sims)
    report["n_add_sims_because_expensive"] = np.sum(sim_expensive)
    report["n_add_sims_because_loose"] = np.sum(sim_loose & (~sim_tiny))

    A, did_refine = execute.ada_refine(A, which_refine)
    report["n_refined"] = np.sum(which_refine)
    report["n_refined_because_expensive"] = np.sum(hob_expensive)
    report["n_refined_because_loose"] = np.sum(hob_loose & (~hob_tiny))
    # elif np.sum(more_sims) > 0:
    # A.target_sim_sizes[more_sims] += 5000
    if np.sum(which_refine) == 0 and np.sum(more_sims) == 0:
        print("done after", ada_i, "iterations")
        break

    report["simulation_runtime"] = f"{sim_runtime:.2f}s"
    report["iteration_runtime"] = f"{time.time() - start:.2f}s"
    rprint(report)
    report_history.append(report)

# plt.show()
```

```python
n_arms = 4
n_arm_samples = 100
ys = np.arange(n_arm_samples + 1)
Ygrids = np.stack(np.meshgrid(*[ys] * n_arms, indexing="ij"), axis=-1)
Yravel = Ygrids.reshape((-1, n_arms))

# 2. Sort the grid arms while tracking the sorting order so that we can
# unsort later.
colsortidx = np.argsort(Yravel, axis=-1)
inverse_colsortidx = np.zeros(Yravel.shape, dtype=np.int32)
axis0 = np.arange(Yravel.shape[0])[:, None]
inverse_colsortidx[axis0, colsortidx] = np.arange(n_arms)
Y_colsorted = Yravel[axis0, colsortidx]

# 3. Identify the unique datasets. In a 35^4 grid, this will be about 80k
# datasets instead of 1.7m.
Y_unique, inverse_unique = np.unique(Y_colsorted, axis=0, return_inverse=True)
(n_arm_samples**n_arms), Y_unique.shape
```

```python
%matplotlib widget
plt.figure(figsize=(8, 8))
plt.scatter(A.g.theta_tiles[:, 0], A.g.theta_tiles[:, 1], c=A.hob_upper, s=2)
plt.colorbar()
plt.show()
# plt.figure(figsize=(20,20))
# plt.scatter(A.g.theta_tiles[:,0], A.g.theta_tiles[:, 1], c=np.log10(A.sim_sizes), s=2)
# plt.colorbar()
# plt.show()
```

```python
def optimal_centering(f, p):
    return 1 / (1 + ((1 - f) / f) ** (1 / (p - 1)))
```

```python
optimal_centering(np.linspace(0.001, 1, 100), 1.2)
```

```python
A.sim_sizes.max(), (4.5 / A.g.radii.min()) ** 2
```

```python
(4.5 / A.g.radii.min()) ** 2 * A.sim_sizes.max() / 1e9, A.total_sims / 1e9
```

```python
hob_cost = np.max(A.hob_upper) - np.max(typeI_upper_bound)
sim_cost = np.max(typeI_upper_bound) - np.max(A.typeI_est)
hob_cost, sim_cost
```

```python

```
