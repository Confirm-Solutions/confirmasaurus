```python
import confirm.outlaw.nb_util as nb_util

nb_util.setup_nb()
import jax
import numpy as np
import matplotlib.pyplot as plt
from confirm.lewislib import lewis
import confirm.imprint as ip
```

```python
name = "1d_slice"
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
import os
import time
import modal
import pandas as pd
import confirm.cloud.modal_util as modal_util

stub = modal.Stub("confirm")

img = modal_util.get_image()


@stub.function(
    image=img,
    gpu=modal.gpu.A100(),
    retries=0,
    mounts=modal.create_package_mounts(["confirm"]),
    timeout=60 * 60 * 2,
)
def cloud_validate(g, *, K, params):
    start = time.time()
    rej_df = ip.validate(
        lewis.Lewis45Model,
        g,
        lam=0.06253,
        K=K,
        tile_batch_size=128,
        model_kwargs=params,
    )
    print(time.time() - start)
    return g.add_cols(rej_df)
```

```python
g = ip.cartesian_grid([-1, -2], [1, -1], n=[10, 400]).concat(
    ip.cartesian_grid([-1, -4], [1, -2], n=[10, 400]),
    ip.cartesian_grid([-1, -8], [1, -4], n=[10, 400]),
    ip.cartesian_grid([-1, -10], [1, -8], n=[10, 100]),
)
g.df["theta3"] = g.df["theta1"]
g.df["theta1"] = g.df["theta0"]
g.df["theta2"] = g.df["theta0"]
g.df["radii3"] = g.df["radii1"]
g.df["radii0"] = 0
g.df["radii1"] = 0
g.df["radii2"] = 0
g.df["null_truth0"] = True
g.df["null_truth1"] = True
g.df["null_truth2"] = True
```

```python
g.df.shape
```

```python
# import pandas as pd
# rej_df = pd.read_parquet("./1d_slice/1d_orthogonal_ada.parquet")
# rej_df
# boundcost = rej_df['tie_bound'] - rej_df['tie_cp_bound']
# refine = boundcost > 0.001
# g_refine = g.subset(refine).refine()
# g_new = g_refine.concat(g.subset(~refine))
# new_t3 = np.sort(np.unique(g_new.df['theta3']))
# new_t3.shape[0]
# plt.plot(new_t3, '*')
# plt.show()
```

```python
n_chunks = 50
with stub.run():
    i = 27
    gs = [ip.grid.Grid(d, g.null_hypos) for d in np.array_split(g.df, n_chunks)][i:]
    rej_gs = []
    for rej_g in cloud_validate.map(
        gs, kwargs={"K": int(1.5 * 2**19), "params": params}, order_outputs=True
    ):
        rej_g.df.to_parquet(f"./1d_slice/chunk_{i}.parquet")
        i += 1
        rej_gs.append(rej_g)
```

```python
rej_gs = []
for i in range(n_chunks):
    chunk_g = pd.read_parquet(f"./1d_slice/chunk_{i}.parquet")
    n_rows = chunk_g.shape[0]
    if i > 0:
        rej_gs.append(
            pd.concat(
                (
                    chunk_g.iloc[: n_rows // 2]
                    .reset_index()
                    .drop(["tie_sum", "tie_est", "tie_cp_bound", "tie_bound"], axis=1),
                    chunk_g.iloc[n_rows // 2 :][
                        ["tie_sum", "tie_est", "tie_cp_bound", "tie_bound"]
                    ].reset_index(),
                ),
                axis=1,
            ).drop("index", axis=1)
        )
    else:
        rej_gs.append(chunk_g)
rej_g_final = ip.grid.Grid(rej_gs[0]).concat(*[ip.grid.Grid(g) for g in rej_gs[1:]])
rej_g_final.df.to_parquet("./1d_slice/1d_orthogonal_chunks.parquet")
```

```python
import pandas as pd

rej_df = pd.read_parquet("./1d_slice/1d_orthogonal_chunks.parquet")
rej_df
```

```python
worst_bound = rej_df.loc[rej_df.reset_index().groupby(["theta3"])["tie_bound"].idxmax()]
worst_bound
```

```python
plt.scatter(rej_df[""])
```

```python
# worst_tie = rej_df["tie_est"].values.reshape((100, 20)).max(axis=1)
# worst_bound = rej_df["tie_bound"].values.reshape((100, 20)).max(axis=1)
plt.plot(
    worst_bound["theta3"],
    100 * worst_bound["tie_est"],
    "k-",
    label="Type I Error",
)
plt.plot(
    worst_bound["theta3"],
    100 * worst_bound["tie_bound"],
    "r-",
    label="Tilt-Bound",
)
plt.legend()
plt.xlabel(r"$\theta_{3}$")
plt.ylabel(r"Type I Error \%")
plt.title(r"$\max_{\theta_c}(f(\theta_c, \theta_c, \theta_c, \theta_3))$")
plt.savefig("lewis_1d_orthogonal.pdf", bbox_inches="tight")
plt.show()
```
