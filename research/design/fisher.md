```python
from confirm.outlaw.nb_util import setup_nb

setup_nb(autoreload=False)

import matplotlib.pyplot as plt
import scipy.stats
import jax
import jax.numpy as jnp
import numpy as np

import confirm.imprint as ip
import confirm.models.fisher_exact as fisher
```

## Binomial two class

```python
K = 2**12
lam = 0.05
for n in range(3, 15):
    g = ip.cartesian_grid(
        [-3, -3], [3, 3], n=[20, 20], null_hypos=[ip.hypo("theta1 < theta0")]
    )
    rej_df = ip.validate(fisher.FisherExact, g, lam, K=K, model_kwargs=dict(n=n))
    print(f"n_arm_samples={n} max(tie)={rej_df['tie_est'].max():.4f}")
```

```python
K = 2**12
lam = 0.05

for n in range(3, 15):
    g = ip.cartesian_grid(
        [-1, -1], [1, 1], n=[1, 1], null_hypos=[ip.hypo("theta1 < theta0")]
    )
    rej_df = ip.validate(fisher.FisherExact, g, lam, K=K, model_kwargs=dict(n=n))
    print(rej_df["tie_est"].max())
    rej_df = ip.validate(fisher.BoschlooExact, g, lam, K=K, model_kwargs=dict(n=n))
    print(rej_df["tie_est"].max())
```

```python
text = """
0.0
0.015869140625
0.00341796875
0.03369140625
0.009033203125
0.0263671875
0.0205078125
0.031005859375
0.010498046875
0.037353515625
0.0107421875
0.03857421875
0.016845703125
0.04931640625
0.021240234375
0.04345703125
0.033447265625
0.047119140625
0.036376953125
0.044921875
0.018798828125
0.034423828125
0.019287109375
0.047119140625
"""
data = np.concatenate(
    (
        np.arange(3, 15)[:, None],
        np.array([float(f) for f in text.split("\n")[1:-1]]).reshape((-1, 2)),
    ),
    axis=1,
)
for i in range(data.shape[0]):
    print(
        f"n={int(data[i,0])} max(fisher)={data[i, 1]:.4f} max(boschloo)={data[i, 2]:.4f}"
    )
```

```python
model = FisherExact(0, 10, n=10)
np.random.seed(0)
theta = np.random.rand(5, 2)
null_truth = np.ones((5, 1), dtype=bool)
np.testing.assert_allclose(
    model._sim_scipy(model.samples[0:10], theta, null_truth),
    model._sim_jax(model.samples[0:10], theta, null_truth),
)
```

```python
g = ip.cartesian_grid(
    [-1, -1], [1, 1], n=[50, 50], null_hypos=[ip.hypo("theta1 < theta0")]
)
# ip.grid.plot_grid(g)
# plt.show()
```

```python
n = 10
K = 2**12
rej_df = ip.validate(FisherExact, g, 0.0286, K=K, model_kwargs=dict(n_arm_samples=n))
```

```python
import confirm.imprint.summary

ip.summary.summarize_validate(g, rej_df)
```

## Tuning Fisher Exact

```python
n = 15
alpha = 0.05
g = ip.cartesian_grid(
    [-1, -1], [1, 1], n=[4, 4], null_hypos=[ip.hypo("theta1 < theta0")]
)
iter, reports, ada = ip.ada_tune(
    fisher.FisherExact,
    g=g,
    alpha=alpha,
    model_kwargs=dict(n=n),
    grid_target=0.0001,
    bias_target=0.0001,
    std_target=0.0001,
    iter_size=2**11,
    n_K_double=6,
    n_iter=1000,
)
```

```python
g_ada = ip.Grid(ada.db.get_all()).active()
K = 2**14
df = g_ada.df
lamss = df["lams"].min()
rej_df = ip.validate(fisher.FisherExact, g_ada, lamss, K=K, model_kwargs=dict(n=n))
```

```python
tiles = [g_ada.df[f"B_lams{i}"].idxmin() for i in range(50)]
g_critical = g_ada.df.loc[tiles]
```

```python
lamss = df["lams"].min()
lamss
```

```python
rej_df = ip.validate(
    fisher.FisherExact,
    ip.Grid(g_critical),
    lamss,
    K=2**20,
    model_kwargs=dict(n=n),
    tile_batch_size=5,
)
```

```python
rej_df
```

```python
plt.figure(figsize=(10, 5), constrained_layout=True)
plt.subplot(1, 2, 1)
plt.suptitle("$\lambda^{**} = " + f"{lamss:.4f} ~~~~ \\alpha = {alpha}$")
plt.scatter(df["theta0"], df["theta1"], c=df["lams"], vmin=lamss, vmax=lamss + 0.1)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.colorbar(label="$\lambda^*$")

plt.subplot(1, 2, 2)
plt.scatter(df["theta0"], df["theta1"], c=rej_df["tie_bound"], vmin=0, vmax=alpha)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.colorbar(label="$\hat{f}(\lambda^{**})$")
plt.show()
```

```python
import pandas as pd

evolution = pd.DataFrame(reports)
fig, ax = plt.subplots(3, 2, figsize=(8, 12), constrained_layout=True)
plt.subplot(3, 2, 1)
plt.plot(evolution["i"], evolution["bias_tie"], "o-")
plt.xlabel("Iteration")
plt.title(r"$bias(\hat{f}(\lambda^{**}))$")
plt.subplot(3, 2, 2)
plt.plot(evolution["i"], evolution["grid_cost"], "o-")
plt.xlabel("Iteration")
plt.title(r"$\alpha - \alpha_0$")
plt.subplot(3, 2, 3)
plt.plot(evolution["i"], evolution["std_tie"], "o-")
plt.xlabel("Iteration")
plt.title(r"$\sigma_{B}(\hat{f}(\lambda^{**}))$")
plt.subplot(3, 2, 4)
plt.scatter(df["theta0"], df["theta1"], c=df["radii0"])
plt.colorbar()
plt.title("Radius")
plt.subplot(3, 2, 5)
plt.scatter(df["theta0"], df["theta1"], c=df["K"])
plt.colorbar()
plt.title("K")
plt.subplot(3, 2, 6)
plt.scatter(df["theta0"], df["theta1"], c=df["alpha0"])
plt.colorbar()
plt.title("alpha0")
plt.show()
```

```python
lamss
```

```python
def compare_tables(n, lam, lamss):
    successes = np.stack(
        np.meshgrid(np.arange(n + 1), np.arange(n + 1)), axis=-1
    ).reshape(-1, 2)
    possible_datasets = np.concatenate(
        (successes[:, None, :], n - successes[:, None, :]),
        axis=1,
    )

    boschloo = np.array(
        [
            scipy.stats.boschloo_exact(possible_datasets[i], alternative="less").pvalue
            for i in range(possible_datasets.shape[0])
        ]
    )
    barnard = np.array(
        [
            scipy.stats.barnard_exact(possible_datasets[i], alternative="less").pvalue
            for i in range(possible_datasets.shape[0])
        ]
    )
    tuned_fisher = np.array(
        [
            scipy.stats.fisher_exact(possible_datasets[i], alternative="less")[1]
            for i in range(possible_datasets.shape[0])
        ]
    )
    differences = np.where(
        ((boschloo < lam) != (tuned_fisher < lamss - 1e-12))
        | ((barnard < lam) != (tuned_fisher < lamss - 1e-12))
    )[0]
    return (
        possible_datasets[differences],
        boschloo[differences],
        boschloo[differences] < lam,
        barnard[differences],
        barnard[differences] < lam,
        tuned_fisher[differences],
        tuned_fisher[differences] < lamss - 1e-12,
    )


compare_tables(n, alpha, lamss)
```

```python
g_check = ip.cartesian_grid(
    [-1, -1], [1, 1], n=[1, 1], null_hypos=[ip.hypo("theta1 < theta0")]
)
K = 256
fisher_df = ip.validate(fisher.FisherExact, g_check, lam, K=K, model_kwargs=dict(n=n))
boschloo_df = ip.validate(
    fisher.BoschlooExact, g_check, lam, K=K, model_kwargs=dict(n=n)
)
barnard_df = ip.validate(fisher.BarnardExact, g_check, lam, K=K, model_kwargs=dict(n=n))
tuned_fisher_df = ip.validate(
    fisher.FisherExact, g_check, lamss, K=K, model_kwargs=dict(n=n)
)
print(f"n=8, fisher(0.05)={fisher_df['tie_est'].max()}")
print(f"n=8, boschloo(0.05)={boschloo_df['tie_est'].max()}")
print(f"n=8, barnard(0.05)={barnard_df['tie_est'].max()}")
print(f"n=8, tuned_fisher({lamss:5f})={tuned_fisher_df['tie_est'].max()}")
print(
    "max difference boschloo vs tuned fisher: ",
    np.abs(barnard_df["tie_est"] - boschloo_df["tie_est"]).max(),
)
print(
    "max difference barnard vs tuned fisher: ",
    np.abs(barnard_df["tie_est"] - boschloo_df["tie_est"]).max(),
)
```
