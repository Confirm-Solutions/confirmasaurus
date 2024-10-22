# Analyze Upper Bound of Type I Error for Lei Example

```python
%load_ext autoreload
%autoreload 2
```

```python
import jax
import os
import numpy as np
from confirm.imprint import grid
from confirm.lewislib import grid as lewgrid
from confirm.lewislib import lewis
from confirm.imprint import binomial
```

```python
# Configuration used during simulation
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
    "n_pr_sims": 100,
    "n_sig2_sims": 20,
    "batch_size": int(2**20),
    "cache_tables": False,
}
size = 52
n_sim_batches = 500
sim_batch_size = 100
```

```python
# construct Lei object
lei_obj = lewis.Lewis45(**params)
```

```python
# construct the same grid used during simulation
n_arms = params["n_arms"]
lower = np.full(n_arms, -1)
upper = np.full(n_arms, 1)
thetas, radii = lewgrid.make_cartesian_grid_range(
    size=size,
    lower=lower,
    upper=upper,
)
ns = np.concatenate(
    [np.ones(n_arms - 1)[:, None], -np.eye(n_arms - 1)],
    axis=-1,
)
null_hypos = [grid.HyperPlane(n, 0) for n in ns]
gr = grid.build_grid(
    thetas=thetas,
    radii=radii,
    null_hypos=null_hypos,
)
gr = grid.prune(gr)
```

```python
# construct tile informations used during simulation
theta_tiles = gr.thetas[gr.grid_pt_idx]
p_tiles = jax.scipy.special.expit(theta_tiles)
tile_radii = gr.radii[gr.grid_pt_idx]
null_truths = gr.null_truth.astype(bool)
sim_size = 2 * n_sim_batches * sim_batch_size  # 2 instances parallelized
sim_sizes = np.full(gr.n_tiles, sim_size)
```

```python
# get type I sum and score
cwd = "."
data_dir = os.path.join(cwd, "../data")
output_dir = os.path.join(data_dir, "output_1")
typeI_sum = np.loadtxt(os.path.join(output_dir, "typeI_sum.csv"), delimiter=",")
typeI_score = np.loadtxt(os.path.join(output_dir, "typeI_score.csv"), delimiter=",")
output_dir = os.path.join(data_dir, "output_2")
typeI_sum += np.loadtxt(os.path.join(output_dir, "typeI_sum.csv"), delimiter=",")
typeI_score += np.loadtxt(os.path.join(output_dir, "typeI_score.csv"), delimiter=",")
```

```python
delta = 0.025
n_arm_samples = int(lei_obj.unifs_shape()[0])
tile_corners = gr.vertices
```

```python
# construct Holder upper bound
d0, d0u = binomial.zero_order_bound(
    typeI_sum=typeI_sum,
    sim_sizes=sim_sizes,
    delta=delta,
    delta_prop_0to1=1,
)
typeI_bound = d0 + d0u

total_holder = binomial.holder_odi_bound(
    typeI_bound=typeI_bound,
    theta_tiles=theta_tiles,
    tile_corners=tile_corners,
    n_arm_samples=n_arm_samples,
    holderq=16,
)
```

```python
# construct classical upper bound
total, d0, d0u, d1w, d1uw, d2uw = binomial.upper_bound(
    theta_tiles,
    tile_radii,
    gr.vertices,
    sim_sizes,
    n_arm_samples,
    typeI_sum,
    typeI_score,
)
```

```python
# prepare bound components

# classical
bound_components = np.array(
    [
        d0,
        d0u,
        d1w,
        d1uw,
        d2uw,
        total,
    ]
).T

# holder
dummy = np.zeros_like(d0)
bound_components_holder = np.array(
    [
        d0,
        d0u,
        dummy,
        dummy,
        dummy,
        total_holder,
    ]
).T
```

```python
t2_uniques = np.unique(theta_tiles[:, 2])
t3_uniques = np.unique(theta_tiles[:, 3])
t2_uniques, t3_uniques
```

```python
# slice and save P, B
t2 = t2_uniques[25]
t3 = t3_uniques[20]
selection = (theta_tiles[:, 2] == t2) & (theta_tiles[:, 3] == t3)
```

```python
bound_dir = os.path.join(data_dir, "bound")
if not os.path.exists(bound_dir):
    os.makedirs(bound_dir)

np.savetxt(
    f"{bound_dir}/P_lei.csv", theta_tiles[selection, :].T, fmt="%s", delimiter=","
)
np.savetxt(
    f"{bound_dir}/B_lei.csv", bound_components[selection, :], fmt="%s", delimiter=","
)
np.savetxt(
    f"{bound_dir}/B_lei_holder.csv",
    bound_components_holder[selection, :],
    fmt="%s",
    delimiter=",",
)
```
