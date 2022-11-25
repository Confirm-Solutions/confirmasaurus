```python
import confirm.outlaw.nb_util as nb_util

nb_util.setup_nb()

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
import scipy.spatial
import pickle

jax.config.update("jax_platform_name", "gpu")
import confirm.imprint.lewis_drivers as lts
from confirm.imprint import grid
import adastate
import diagnostics
```

```python
from confirm.lewislib import lewis, batch

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
lei_obj = lewis.Lewis45(**params)
```

```python
with open(f"./{name}/data_params.pkl", "rb") as f:
    P, D = pickle.load(f)
with open(f"./{name}/rerun_final.pkl", "rb") as f:
    _, S = pickle.load(f)
# load_iter = "latest"
# S, load_iter, fn = adastate.load(name, load_iter)
```

```python
import criterion

cr = criterion.Criterion(lei_obj, P, S, D)
```

```python
cr.alpha_cost[cr.overall_tile]
```

```python
cr.bias
```

```python
overall_lam = S.orig_lam.min()
```

```python
worst_tile = np.array([0.53, 0.53, 0.53, -1.0])
```

## Tile density

```python
plot_dims = [2, 3]
slc = diagnostics.build_2d_slice(S.g, worst_tile, plot_dims)
```

```python
(2.0 / S.g.radii.min()) ** 4 * 10000 / 900e9
```

```python
S.g.radii[S.g.grid_pt_idx].max()
```

```python
S.g.radii[S.g.grid_pt_idx].min()
```

```python
(2.0 / 0.00048828) ** 4 / S.g.n_tiles / 1e6
```

```python
S.sim_sizes.max()
```

```python
theta_tiles2 = S.g.theta_tiles.copy()
theta_tiles2[:, 2] = S.g.theta_tiles[:, 3]
theta_tiles2[:, 3] = S.g.theta_tiles[:, 2]
sym_tiles1 = np.concatenate((S.g.theta_tiles, theta_tiles2))

theta_tiles3 = sym_tiles1.copy()
theta_tiles3[:, 1] = sym_tiles1[:, 2]
theta_tiles3[:, 2] = sym_tiles1[:, 1]
all_tiles = np.concatenate((sym_tiles1, theta_tiles3))
```

```python
all_lam = np.concatenate((S.orig_lam, S.orig_lam, S.orig_lam, S.orig_lam))
```

```python
tree = scipy.spatial.KDTree(all_tiles)
```

```python
nearby = tree.query_ball_point(slc.reshape((-1, 4)), 0.04)
nearby_count = [len(n) for n in nearby]
```

```python
dist1, idx1 = tree.query(slc.reshape((-1, 4)), k=1)
```

```python
dist, idx = tree.query(slc.reshape((-1, 4)), k=10)
```

```python
1.0 / dist1
```

```python
dist.shape
```

```python
x = slc[..., plot_dims[0]]
y = slc[..., plot_dims[1]]
z = dist.mean(axis=1).reshape(slc.shape[:2])
z = np.tril(z)
z = z + z.T - np.diag(np.diag(z))
z = (0.01**4) / (z**4)
z = np.log10(z)
levels = np.linspace(-2, 4, 7)
cntf = plt.contourf(x, y, z, levels=levels, cmap="viridis", extend="min")
plt.contour(
    x, y, z, levels=levels, colors="k", linestyles="-", linewidths=0.5, extend="min"
)
cbar = plt.colorbar(cntf)
plt.xlabel(f"$\\theta_{plot_dims[0]}$")
plt.ylabel(f"$\\theta_{plot_dims[1]}$")
plt.show()
```

```python
x = slc[..., plot_dims[0]]
y = slc[..., plot_dims[1]]
z = np.array(nearby_count).reshape(slc.shape[:2])
# z[z == 0] = z.T[z == 0]
z = np.tril(z)
z = z + z.T - np.diag(np.diag(z))
z = np.log10(z)
levels = np.linspace(0, 5, 11)
plt.title("$\log_{10}$(number of nearby tiles)")
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
cbar = plt.colorbar(cntf, ticks=np.arange(6))
cbar.ax.set_yticklabels(["1", "10", "$10^2$", "$10^3$", "$10^4$", "$10^5$"])
plt.xlabel(f"$\\theta_{plot_dims[0]}$")
plt.ylabel(f"$\\theta_{plot_dims[1]}$")
plt.show()
```

## Type I error calculate

```python
import confirm.imprint.lewis_drivers as ld
```

```python
plot_dims = [2, 3]
slc = diagnostics.build_2d_slice(S.g, worst_tile, plot_dims)
slc_ravel = slc.reshape((-1, S.g.d))
nx, ny, _ = slc.shape
eval_pts = slc.reshape((-1, 4))
null_truth = np.array([eval_pts.dot(H.n) - H.c >= 0 for H in S.g.null_hypos]).T
```

```python
K = 32768
K * eval_pts.shape[0] * 5e-7
```

```python
# overall_lam = 0.0625298
overall_lam = 0.05633
# typeI_sum = ld.rej_runner(
#     lei_obj,
#     np.full(eval_pts.shape[0], K),
#     overall_lam,
#     eval_pts,
#     null_truth,
#     D.unifs,
#     D.unifs_order,
# )
```

```python
lamstar = ld.bootstrap_tune_runner(
    lei_obj,
    np.full(eval_pts.shape[0], 2**15),
    np.full(eval_pts.shape[0], 0.025),
    eval_pts,
    null_truth,
    D.unifs,
    D.bootstrap_idxs,
    D.unifs_order,
)
```

```python
with open("4d_full/plot.pkl", "rb") as f:
    slc_load, typeI_sum = pickle.load(f)
np.testing.assert_allclose(slc, slc_load)
with open("4d_full/plot.pkl", "wb") as f:
    pickle.dump((slc, typeI_sum), f)
```

```python
# with open("4d_full/plot_lamstar.pkl", "rb") as f:
#     slc, lamstar = pickle.load(f)
with open("4d_full/plot_lamstar.pkl", "wb") as f:
    pickle.dump((slc, lamstar), f)
```

```python
with open("4d_full/plot_all.pkl", "wb") as f:
    pickle.dump((slc, typeI_err, lamstar, nearby_count), f)
```

```python
typeI_err = typeI_sum / K
typeI_err[np.all(~null_truth, axis=1)] = np.nan
import confirm.imprint.binomial as binomial

delta = 0.01
typeI_err, typeI_CI = binomial.zero_order_bound(typeI_sum, K, delta, 1.0)
typeI_bound = typeI_err + typeI_CI
```

```python
import confirm.imprint.bound.binomial as tiltbound

n_arm_samples = lei_obj.n_arm_samples
theta0 = eval_pts
v = eval_pts - theta0
```

```python
fwd_solver = tiltbound.ForwardQCPSolver(n=lei_obj.n_arm_samples)


def forward_bound(theta0, vertices, f0):
    v = vertices - theta0
    q_opt = fwd_solver.solve(theta0, v, f0)
    return tiltbound.tilt_bound_fwd_tile(q_opt, n_arm_samples, theta0, v, f0)


bound = jax.jit(jax.vmap(forward_bound))(theta0, eval_pts, typeI_bound)
```

```python
# step 1: evaluate the field of interest. if it's lambda*, we already have what
# we need. if it's TIE, we need to calculate it for the relevant tiles.
```

```python
unplot_dims = list(set(range(S.g.d)) - set(plot_dims))
```

```python
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
## See ../paper_figures/lewis.ipynb for the final paper plots!
```
