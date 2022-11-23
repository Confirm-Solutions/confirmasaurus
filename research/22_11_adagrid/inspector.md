```python
import confirm.outlaw.nb_util as nb_util

nb_util.setup_nb()

import pickle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

# Run on CPU because a concurrent process is probably running on GPU.
jax.config.update("jax_platform_name", "cpu")

import confirm.imprint.lewis_drivers as lts
from confirm.lewislib import lewis

import adastate
from criterion import Criterion
from diagnostics import lamstar_histogram
```

```python
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
load_iter = "latest"
S, load_iter, fn = adastate.load(name, load_iter)
```

```python
plt.hist(S.B_lam.min(axis=0))
plt.show()
```

```python
cr = Criterion(lei_obj, P, S, D)
assert S.twb_max_lam[cr.twb_worst_tile] == np.min(S.twb_max_lam)
assert S.twb_min_lam[cr.twb_worst_tile] == np.min(S.twb_min_lam[cr.ties])
```

```python
Blamsort = S.B_lam.argsort(axis=0)
```

```python
origlamsort = S.orig_lam.argsort()
```

```python
Blamsort[0]
```

```python
import confirm.imprint.lewis_drivers as ld

for i in [0, 1, 10, 100, 200, 300, 500, 750, 1000, 5000, 10000, 100000]:
    B_lamss_idx = Blamsort[i, :]
    B_lamss = S.B_lam[B_lamss_idx, np.arange(S.B_lam.shape[1])]
    overall_tile = origlamsort[i]
    overall_lam = S.orig_lam[overall_tile]
    bootstrap_min_lams = np.concatenate(([overall_lam], B_lamss))
    overall_stats = ld.one_stat(
        lei_obj,
        S.g.theta_tiles[overall_tile],
        S.g.null_truth[overall_tile],
        S.sim_sizes[overall_tile],
        D.unifs,
        D.unifs_order,
    )
    overall_typeI_sum = (overall_stats[None, :] < B_lamss[:, None]).sum(axis=1)
    bias = (overall_typeI_sum[0] - overall_typeI_sum[1:].mean()) / S.sim_sizes[
        overall_tile
    ]
    print(f"index={i} bias={bias:5f}")
```

```python
overall_typeI_sum = (overall_stats[None, :] < bootstrap_min_lams[:, None]).sum(axis=1)
bias = (overall_typeI_sum[0] - overall_typeI_sum[1:].mean()) / S.sim_sizes[overall_tile]
```

```python
cr.bias
```

```python
tie = cr.overall_typeI_sum / S.sim_sizes[cr.overall_tile]
tie[0] - np.mean(tie[1:])
```

```python
biases = [tie[i] - np.mean(np.delete(tie, i)) for i in range(1, len(tie))]
plt.hist(biases)
plt.show()
```

```python
plt.hist()
plt.show()
```

```python
idxs = cr.dangerous[:10]
alpha0_new = adastate.AdaRunner(P, lei_obj).batched_invert_bound(
    S.g.theta_tiles[idxs], S.g.vertices(idxs)
)
alpha0_new
```

```python
alpha0_new, alpha0_new - S.alpha0[idxs]
```

## 11/1/2022

```python
import pandas as pd
```

```python
# orderer = combined_mean_idx + inflation * (combined_min_idx - combined_mean_idx)
# orderer = S.twb_mean_lam + inflation * (S.twb_min_lam - S.twb_mean_lam)
# orderer[S.twb_mean_lam >= 0.3] = 1.0
# def explore_orderer():
#     sorted_ordering = np.argsort(orderer)
#     sorted_orderer = orderer[sorted_ordering]
#     print(S.db.data[sorted_ordering[:10], S.db.slices['twb_min_lam']])
#     print(S.db.data[sorted_ordering[:10], S.db.slices['twb_mean_lam']])
#     print(S.db.data[sorted_ordering[:1000000], S.db.slices['twb_min_lam']].max())
```

```python
from IPython.display import display


def tile_report(idxs):
    return pd.DataFrame(
        dict(
            order_idx=np.searchsorted(cr.sorted_orderer, cr.orderer[idxs]),
            twb_min_lam_idx=np.searchsorted(cr.sorted_orderer, S.twb_min_lam[idxs]),
            orderer=cr.orderer[idxs],
            B_lams_min=S.B_lam[idxs].min(axis=1),
            twb_min_lam=S.twb_min_lam[idxs],
            twb_mean_lam=S.twb_mean_lam[idxs],
            twb_max_lam=S.twb_max_lam[idxs],
            orig_lam=S.orig_lam[idxs],
            sim_size=S.sim_sizes[idxs],
            alpha0=S.alpha0[idxs],
            alpha_cost=cr.alpha_cost[idxs],
        )
    )


rpt = tile_report(cr.B_lamss_idx)
rpt["B_lamss"] = cr.B_lamss
rpt.sort_values("B_lamss")
```

```python
display(tile_report([cr.twb_worst_tile]))
cr.twb_worst_tile_lam_min, cr.twb_worst_tile_lam_mean, cr.twb_worst_tile_lam_max
```

```python
tile_report(cr.dangerous)
```

```python
tile_report(cr.refine_dangerous)
```

```python
overall_rpt = tile_report(S.orig_lam.argsort()[:1000])
overall_rpt
```

```python
print("overall_lam", cr.overall_lam)
B_min = S.B_lam.min(axis=1)
bias_bad = B_min < cr.overall_lam
print("n bias bad", np.sum(bias_bad))
n_critical = np.sum((S.orig_lam < cr.overall_lam + 0.01))
n_loose = np.sum(
    (S.orig_lam < cr.overall_lam + 0.01) & (P.alpha_target - S.alpha0 > P.grid_target)
)
print(f"number of tiles near critical: {n_critical}")
print(f"    and with loose bounds {n_loose}")
# for i in range(10):
#     dangerous = np.sum(cr.inflated_min_lam[bias_bad] < cr.overall_lam)
#     collateral = np.sum(cr.inflated_min_lam < cr.overall_lam)
#     print(f'inflation factor {i}')
#     print(f'    dangerous tiles caught: {dangerous}')
#     print(f'    collateral tiles caught: {collateral}')

print("lambda**B", cr.B_lamss)
total_effort = np.sum(S.sim_sizes)
for K in np.unique(S.sim_sizes):
    sel = S.sim_sizes == K
    count = np.sum(sel)
    print(f"K={K}:")
    print(f"    count={count / 1e6:.3f}m")
    print(f"    lambda**B[K]={S.B_lam[sel].min(axis=0)}")
    print(f"    min lambda*B[K]={np.min(S.B_lam[sel].min(axis=1)):.4f}")
    print(f"    min lambda*b[K]={np.min(S.twb_min_lam[sel]):.4f}")
    effort = K * count / total_effort
    print(f"    % effort={100 * effort:.4f}")
```

```python
plt.figure(figsize=(10, 10), constrained_layout=True)
plt.subplot(2, 2, 1)
plt.title("$min(\lambda^*_B)$")
lamstar_histogram(S.B_lam.min(axis=1), S.sim_sizes)
for i, (field, title) in enumerate(
    [
        (S.orig_lam, "$\lambda^{*}$"),
        (S.twb_min_lam, "$min(\lambda^*_b)$"),
        (S.twb_mean_lam, "$mean(\lambda^*_b)$"),
    ]
):
    plt.subplot(2, 2, i + 2)
    plt.title(title)
    lamstar_histogram(field, S.sim_sizes)
plt.show()
```

## Scratch


## Resimulation

```python
import pandas as pd

friends = np.where(bootstrap_cvs[:, 0] < 0.045)[0]
print(pd.DataFrame(sim_sizes[friends]).describe())
print(pd.DataFrame(pointwise_target_alpha[friends]).describe())
```

```python
seed = 0
src_key = jax.random.PRNGKey(seed)
key1, key2, key3 = jax.random.split(src_key, 3)

unifs = jax.random.uniform(
    key=key1, shape=(adap.max_sim_size,) + lei_obj.unifs_shape(), dtype=jnp.float32
)
unifs_order = jnp.arange(0, unifs.shape[1])
nB_global = 30
nB_tile = 40
bootstrap_idxs = {
    K: jnp.concatenate(
        (
            jnp.arange(K)[None, :],
            jax.random.choice(key2, K, shape=(nB_global, K), replace=True),
            jax.random.choice(key3, K, shape=(nB_tile, K), replace=True),
        )
    ).astype(jnp.int32)
    for K in (adap.init_K * 2 ** np.arange(0, adap.n_sim_double + 1))
}
```

```python
print("hi")
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
    grid_batch_size=4,
)
```

```python
stats = np.random.rand(3, 1000)
```

```python
from confirm.lewislib import batch

grid_batch_size = 4


def printer(x, y, z):
    print(x.shape, y.shape, z.shape)
    return 0


tunev = jax.jit(
    jax.vmap(jax.vmap(lts.tune, in_axes=(None, 0, None)), in_axes=(0, None, 0))
)
batched_tune = batch.batch(
    batch.batch(tunev, 10, in_axes=(None, 0, None), out_axes=(1,)),
    grid_batch_size,
    in_axes=(0, None, 0),
)
batched_tune(stats, bootstrap_idxs[1000], np.array([0.025, 0.025, 0.025])).shape
```

```python
bootstrap_idxs[1000].shape
```

```python
batch.batch(lts.tunev, 10, in_axes=(None, 0, None))(
    stats[0], bootstrap_idxs[1000], 0.025
).shape
```

```python
tunev(stats, bootstrap_idxs[1000], np.full(3, 0.025)).shape
```

```python
bootstrap_idxs[1000].shape
```

## Scratch

```python
# typeI_sum = batched_rej(
#     sim_sizes,
#     (np.full(sim_sizes.shape[0], overall_cv),
#     g.theta_tiles,
#     g.null_truth,),
#     unifs,
#     unifs_order,
# )

# savedata = [
#     g,
#     sim_sizes,
#     bootstrap_cvs,
#     typeI_sum,
#     hob_upper,
#     pointwise_target_alpha
# ]
# with open(f"{name}/final.pkl", "wb") as f:
#     pickle.dump(savedata, f)

# # Calculate actual type I errors?
# typeI_est, typeI_CI = binomial.zero_order_bound(
#     typeI_sum, sim_sizes, delta_validate, 1.0
# )
# typeI_bound = typeI_est + typeI_CI

# hob_upper = binomial.holder_odi_bound(
#     typeI_bound, g.theta_tiles, g.vertices, n_arm_samples, holderq
# )
# sim_cost = typeI_CI
# hob_empirical_cost = hob_upper - typeI_bound
# worst_idx = np.argmax(typeI_est)
# worst_tile = g.theta_tiles[worst_idx]
# typeI_est[worst_idx], worst_tile
# worst_cv_idx = np.argmin(sim_cvs)
# typeI_est[worst_cv_idx], sim_cvs[worst_cv_idx], g.theta_tiles[worst_cv_idx], pointwise_target_alpha[worst_cv_idx]
# plt.hist(typeI_est, bins=np.linspace(0.02,0.025, 100))
# plt.show()

# theta_0 = np.array([-1.0, -1.0, -1.0])      # sim point
# v = 0.1 * np.ones(theta_0.shape[0])     # displacement
# f0 = 0.01                               # Type I Error at theta_0
# fwd_solver = ehbound.ForwardQCPSolver(n=n_arm_samples)
# q_opt = fwd_solver.solve(theta_0=theta_0, v=v, a=f0) # optimal q
# ehbound.q_holder_bound_fwd(q_opt, n_arm_samples, theta_0, v, f0)
```
