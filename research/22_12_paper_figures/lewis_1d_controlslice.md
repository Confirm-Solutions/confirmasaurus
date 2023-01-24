```python
import imprint.nb_util as nb_util

nb_util.setup_nb()
import jax
import numpy as np
import matplotlib.pyplot as plt
from confirm.lewislib import lewis
import imprint as ip
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
bad_arm = -1.0
control = np.linspace(-1.0, 1.0, 2000)
theta = np.stack((control, control, control, np.full_like(control, bad_arm)), axis=1)
radii = theta.copy()
radii[:, :3] = (control[1] - control[0]) * 0.5
radii[:, -1] = 0
g_raw = ip.init_grid(theta, radii, 0)
g_raw.df["null_truth0"] = True
g_raw.df["null_truth1"] = True
g_raw.df["null_truth2"] = True
g = g_raw
# .add_null_hypos([
#     ip.hypo('theta0 > theta1'), ip.hypo('theta0 > theta2'), ip.hypo('theta0 > theta3')
# ])
# g = g_raw.subset(g_raw.df['null_truth0'] & g_raw.df['null_truth1'] & g_raw.df['null_truth2'])
```

```python
# rej_df = ip.validate(
#     lewis.Lewis45Model,
#     g,
#     lam=0.06253,
#     K=2**12,
#     tile_batch_size=256,
#     model_kwargs=params,
# )
# plt.plot(g.df['theta0'], rej_df['tie_est'], '.')
# plt.show()
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
    timeout=60 * 30,
)
def cloud_validate(g, *, K, params):
    start = time.time()
    rej_df = ip.validate(
        lewis.Lewis45Model,
        g,
        lam=0.06253,
        K=K,
        tile_batch_size=256,
        model_kwargs=params,
    )
    print(time.time() - start)
    return rej_df


def parallel_validate(g, *, n_workers, K, params):
    gs = [ip.grid.Grid(d, g.null_hypos) for d in np.array_split(g.df, n_workers)]
    with stub.run():
        rej_dfs = list(
            cloud_validate.map(
                gs, kwargs={"K": K, "params": params}, order_outputs=True
            )
        )
    return pd.concat(rej_dfs)
```

## [-1, 1] with $\theta_4 = -1$

```python
# rej_df = parallel_validate(g, n_workers=2, K=2**20, params=params)
with stub.run():
    rej_df = cloud_validate(g, K=2**20, params=params)
```

```python
rej_df.to_parquet("./1d_slice/1d_slice.parquet")
```

```python
import pandas as pd

rej_df = pd.read_parquet("./1d_slice/1d_slice.parquet")
```

```python
max_tile = g.df.loc[rej_df["tie_bound"].idxmax()]
```

```python
max_tile
```

```python
plt.plot(g.df["theta0"], 100 * rej_df["tie_est"], "k-", label="Type I Error")
# plt.plot(g.df["theta0"], 100 * rej_df["tie_cp_bound"], "b-", label="Clopper-Pearson")
plt.plot(g.df["theta0"], 100 * rej_df["tie_bound"], "k--", label="Tilt-Bound")
plt.xlim([-1, 1])
plt.ylim([0, 2.5])
plt.xlabel("$\\theta_{c}$")
plt.ylabel("Type I Error \%")
plt.legend()
plt.title("Slice at $(\\theta_{c}, \\theta_{c}, \\theta_{c}, -1)$")
plt.savefig("lewis_1d_slice.pdf", bbox_inches="tight")
plt.show()
```

## Orthogonal slice

```python
g = ip.cartesian_grid([-1, -10], [1, -1], n=[10, 200])
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
g.df["theta0"]
```

```python
# bad_arm = np.linspace(-5.0, -1.0, 2000)
# control = np.full_like(bad_arm, 0.492746)
# theta = np.stack(
#     (control, control, control, np.full_like(control, bad_arm)), axis=1
# )
# radii = theta.copy()
# radii[:,:3] = 0.0
# radii[:,-1] = (bad_arm[1] - bad_arm[0]) * 0.5
# # radii[:,-1] = 0
# g_raw = ip.init_grid(theta, radii)
# g_raw.df['null_truth0'] = True
# g_raw.df['null_truth1'] = True
# g_raw.df['null_truth2'] = True
# g = g_raw
```

```python
with stub.run():
    rej_df = cloud_validate(g, K=2**20, params=params)
```

```python
rej_df.to_parquet("./1d_slice/1d_orthogonal_wider.parquet")
```

```python
import pandas as pd

rej_df = pd.read_parquet("./1d_slice/1d_orthogonal_wider.parquet")
```

```python
rej_df
```

```python

```

```python
worst_tie = rej_df["tie_est"].values.reshape((100, 20)).max(axis=1)
worst_bound = rej_df["tie_bound"].values.reshape((100, 20)).max(axis=1)
plt.plot(
    g.df["theta3"].values.reshape((100, 20))[:, 0],
    100 * worst_tie,
    "k-",
    label="Type I Error",
)
plt.plot(
    g.df["theta3"].values.reshape((100, 20))[:, 0],
    100 * worst_bound,
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

```python
plt.plot(g.df["theta3"], 100 * rej_df["tie_est"], "k-", label="TIE")
plt.plot(g.df["theta3"], 100 * rej_df["tie_cp_bound"], "b-", label="Clopper-Pearson")
plt.plot(g.df["theta3"], 100 * rej_df["tie_bound"], "r-", label="bound")
# plt.xlim([-1, 1])
# plt.ylim([0, 2.5])
plt.xlabel("$\\theta_{\mathrm{control}}$")
plt.ylabel("Type I Error %")
plt.legend()
plt.title(
    "Slice at $(\\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, -1)$"
)
plt.savefig("lewis_1d_orthogonal.pdf", bbox_inches="tight")
plt.show()
```

## Double figure

```python
bad_arm = -1.0
control = np.linspace(-1.0, 1.0, 2000)
theta = np.stack((control, control, control, np.full_like(control, bad_arm)), axis=1)
radii = theta.copy()
radii[:, :3] = (control[1] - control[0]) * 0.5
radii[:, -1] = 0
g_raw = ip.init_grid(theta, radii)
g_raw.df["null_truth0"] = True
g_raw.df["null_truth1"] = True
g_raw.df["null_truth2"] = True
g1d = g_raw
rej_df1d = pd.read_parquet("./1d_slice/1d_slice.parquet")

g = ip.cartesian_grid([-1, -5], [1, -1], n=[20, 100])
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
gortho = g
rej_dfortho = pd.read_parquet("./1d_slice/1d_orthogonal.parquet")
```

```python
worst_tie = rej_dfortho["tie_est"].values.reshape((100, 20)).max(axis=1)
plt.figure(figsize=(8, 4), constrained_layout=True)
plt.subplot(1, 2, 1)
plt.plot(g1d.df["theta0"], 100 * rej_df1d["tie_est"], "k-", label="Type I Error")
# plt.plot(g.df["theta0"], 100 * rej_df["tie_cp_bound"], "b-", label="Clopper-Pearson")
plt.plot(g1d.df["theta0"], 100 * rej_df1d["tie_bound"], "r-", label="Tilt-Bound")
plt.xlim([-1, 1])
plt.ylim([0, 2.5])
plt.xlabel("$\\theta_{c}$")
plt.ylabel("Type I Error \%")
plt.legend()
plt.title("Slice at $(\\theta_{c}, \\theta_{c}, \\theta_{c}, -1)$")

plt.subplot(1, 2, 2)
plt.plot(gortho.df["theta3"].values.reshape((100, 20))[:, 0], 100 * worst_tie, "k-")
plt.xlabel(r"$\theta_{4}$")
plt.ylabel(r"Type I Error \%")
plt.title(r"$\max_{\theta_c}(f(\theta_c, \theta_c, \theta_c, \theta_4))$")
plt.savefig("lewis_1d_combined.pdf", bbox_inches="tight")
plt.show()
```

## Far into the distance

```python
g = ip.cartesian_grid([-1, -5], [1, -1], n=[20, 100])
g = ip.cartesian_grid([-1, -500], [1, -100], n=[20, 10])
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
with stub.run():
    rej_df = cloud_validate(g, K=2**17, params=params)
```

```python
rej_df["tie_est"].values.reshape((10, 20)).max(axis=1)
```

## [-5, 5] with $\theta_4 = -1$

```python
g = ip.cartesian_grid([-5], [5], n=[500])
with stub.run():
    rej_df = cloud_validate(g, 2**17, params)
```

```python
plt.plot(g.df["theta0"], rej_df["tie_est"], "k-", label="TIE")
plt.plot(g.df["theta0"], rej_df["tie_bound"], "r-", label="bound")
plt.xlabel("$\\theta_{\mathrm{control}}$")
plt.ylabel("Type I Error %")
plt.legend()
plt.title(
    "Slice at $(\\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, -1)$"
)
plt.show()
```

## [-1, 1] with $\theta_4 = -5$

```python
g = ip.cartesian_grid([-3], [1], n=[1000])
params5 = params.copy()
params5["bad_arm"] = -5.0
with stub.run():
    rej_df = cloud_validate(g, 2**20, params5)
```

```python
plt.plot(g.df["theta0"], rej_df["tie_est"], "k-", label="TIE")
plt.plot(g.df["theta0"], rej_df["tie_bound"], "r-", label="bound")
plt.xlabel("$\\theta_{\mathrm{control}}$")
plt.ylabel("Type I Error %")
plt.legend()
plt.title(
    "Slice at $(\\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, -5)$"
)
plt.show()
```

```python

```

## Testing with AWS Batch instead of Modal.

```python
import jax

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
    "bad_arm": -1.0,
}
```

```python
import confirm
import confirm.cloud.awsbatch as awsbatch
import boto3
import pickle


@awsbatch.include_package(confirm)
def job():
    from confirm.lewislib import lewis

    class Model1D:
        def __init__(self, seed, max_K, **kwargs):
            self.model = lewis.Lewis45Model(
                seed, max_K, **{k: v for k, v in kwargs.items() if k != "bad_arm"}
            )
            self.bad_arm = kwargs["bad_arm"]
            self.family = "binomial"
            self.family_params = {"n": int(self.model.lewis45.unifs_shape()[0])}

        def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
            control = theta[:, 0]
            theta = np.stack(
                (control, control, control, np.full_like(control, self.bad_arm)), axis=1
            )
            null_truth = np.full((theta.shape[0], 3), True)
            out = self.model.sim_batch(begin_sim, end_sim, theta, null_truth)
            return out

    import confirm.imprint as ip

    g = ip.cartesian_grid([-1], [1], n=[500])
    K = 2**17

    start = time.time()
    print("starting")
    rej_df = ip.validate(
        Model1D,
        g,
        lam=0.06253,
        K=K,
        tile_batch_size=256,
        model_kwargs=params,
    )
    print(time.time() - start)
    boto3.resource("s3").Object("imprint-dump", "result.pkl").put(
        Body=pickle.dumps(rej_df)
    )
```

```python
awsbatch.local_test(job)
```

```python
response, bucket, filename = awsbatch.remote_run(job, cpus=4, memory=2**15, gpu=True)
```

```python
boto3.resource("s3").Bucket("imprint-dump").download_file("result.pkl", "result.pkl")
g = ip.cartesian_grid([-1], [1], n=[500])
with open("result.pkl", "rb") as f:
    rej_df = pickle.load(f)
```

```python
plt.plot(g.df["theta0"], rej_df["tie_est"], "k-", label="TIE")
plt.plot(g.df["theta0"], rej_df["tie_bound"], "r-", label="bound")
plt.xlabel("$\\theta_{\mathrm{control}}$")
plt.ylabel("Type I Error %")
plt.legend()
plt.title(
    "Slice at $(\\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, \\theta_{\mathrm{control}}, -1)$"
)
plt.show()
```
