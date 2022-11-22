```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()

import matplotlib.pyplot as plt
import scipy.stats
import jax
import jax.numpy as jnp
import numpy as np

import confirm.imprint as ip
from confirm.models.binom1d import Binom1D
```

## Binomial

```python
g = ip.cartesian_grid([-1], [1], n=[100], null_hypos=[ip.hypo("x < 0")])
tune_df = ip.tune(Binom1D, g, model_kwargs=dict(n_arm_samples=10))
lam = tune_df["lams"].min()
print(lam)

K = 8192
rej_df = ip.validate(Binom1D, g, lam, K=K, model_kwargs=dict(n_arm_samples=10))
p = scipy.special.expit(g.get_theta()[:, 0])
true_err = scipy.stats.binom.cdf(lam * 10, 10, 1 - p)

plt.plot(p, rej_df["tie_est"], "bo", markersize=2)
plt.plot(p, rej_df["tie_cp_bound"], "ko", markersize=2)
plt.plot(p, rej_df["tie_bound"], "ro", markersize=2)
plt.plot(p, true_err, "r-o", markersize=2)
plt.show()
plt.plot(p, tune_df["lams"])
plt.show()
```

```python
iter, reports, ada = ip.ada_tune(Binom1D, g, model_kwargs=dict(n_arm_samples=10))
```

## Binomial two class

```python

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

```python
g = ip.cartesian_grid(
    [-1, -1], [1, 1], n=[4, 4], null_hypos=[ip.hypo("theta1 < theta0")]
)
iter, reports, ada = ip.ada_tune(
    FisherExact, g, alpha=0.1, model_kwargs=dict(n_arm_samples=n)
)
```

```python
all_df = ada.tiledb.get_all()
all_df.to_parquet("fisher_save.parquet")
```

```python
ga = ip.Grid(all_df).active()
active_df = ga.df
lamss = active_df["lams"].min()
rej_df = ip.validate(FisherExact, ga, lamss, K=K, model_kwargs=dict(n_arm_samples=n))
```

```python
plt.figure(figsize=(10, 5), constrained_layout=True)
plt.subplot(1, 2, 1)
plt.suptitle("$\lambda^{**} = " + f"{lamss:.4f} ~~~~ \\alpha = 0.1$")
plt.scatter(
    active_df["theta0"], active_df["theta1"], c=active_df["lams"], vmin=0, vmax=0.1
)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.colorbar(label="$\lambda^*$")

plt.subplot(1, 2, 2)
plt.scatter(
    active_df["theta0"], active_df["theta1"], c=rej_df["tie_bound"], vmin=0, vmax=0.1
)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.colorbar(label="$\hat{f}(\lambda^{**})$")
plt.show()
```

```python
import pandas as pd

evolution = pd.DataFrame(reports)
# Figure plotting bias, grid_cost and std_tie
fig, ax = plt.subplots(3, 2, figsize=(8, 12), constrained_layout=True)
ax[0][0].plot(evolution["i"], evolution["bias_tie"], "o-")
ax[0][0].set_xlabel("Iteration")
ax[0][0].set_title(r"$bias(\hat{f}(\lambda^{**}))$")
ax[0][1].plot(evolution["i"], evolution["grid_cost"], "o-")
ax[0][1].set_xlabel("Iteration")
ax[0][1].set_title(r"$\alpha - \alpha_0$")
ax[1][0].plot(evolution["i"], evolution["std_tie"], "o-")
ax[1][0].set_xlabel("Iteration")
ax[1][0].set_title(r"$\sigma_{B}(\hat{f}(\lambda^{**}))$")
sc11 = ax[1][1].scatter(active_df["theta0"], active_df["theta1"], c=active_df["radii0"])
plt.colorbar(sc11)
ax[1][1].set_title("Radius")
sc20 = ax[2][0].scatter(active_df["theta0"], active_df["theta1"], c=active_df["K"])
plt.colorbar(sc20)
ax[2][0].set_title("K")
sc21 = ax[2][1].scatter(active_df["theta0"], active_df["theta1"], c=active_df["alpha0"])
plt.colorbar(sc21)
ax[2][1].set_title("alpha0")
plt.show()
```

```python

```
