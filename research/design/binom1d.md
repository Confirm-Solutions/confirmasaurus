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
iter, reports, ada = ip.ada_tune(Binom1D, g=g, model_kwargs=dict(n_arm_samples=10))
```
