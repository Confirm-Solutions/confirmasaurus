```python
import outlaw.nb_util as nb_util

nb_util.setup_nb()
```

```python
import jax
import jax.numpy as jnp
import numpy as np
import time
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import logit, expit

from outlaw import FullLaplace
import outlaw.quad as quad
from outlaw.berry_model import berry_model, fast_berry
```

```python
n_arms = 4
fl = FullLaplace(berry_model(n_arms), "sig2", np.zeros((4, 2)))
sig2_rule = quad.log_gauss_rule(15, 1e-2, 1e2)
fl = fast_berry(sig2_rule.pts, n_arms)
dtype = np.float32
# for N in 2 ** np.array([4, 9, 14, 16]):
for N in 2 ** np.array([15]):
    y = scipy.stats.binom.rvs(35, 0.3, size=(N, n_arms))
    n = np.full_like(y, 35)
    D = np.stack((y, n), axis=-1).astype(dtype)
    x0 = np.zeros((D.shape[0], sig2_rule.pts.shape[0], 4), dtype=dtype)
    f = lambda: fl(
        dict(sig2=sig2_rule.pts.astype(dtype), theta=None), D, x0, should_batch=False
    )
    f()
    for i in range(20):
        start = time.time()
        post, x_max, hess, iters = f()
        end = time.time()
        print(
            f"{N} datasets, {(end - start) / N * 1e6:.3f} us per dataset, {end - start:.2f}s total"
        )
```

```python
cov = np.full((15, n_arms, n_arms), 100.0)
arms = np.arange(n_arms)
cov[:, arms, arms] += sig2_rule.pts[:, None]
neg_precQ = -np.linalg.inv(cov)
np.diagonal(neg_precQ, axis1=1, axis2=2)
```

```python
theta = np.array([-7.21806323, -2.096633, -0.21382083, -0.06924673])
mu_0 = -1.34
theta_m0 = theta - mu_0
exp_theta_adj = jnp.exp(theta + logit(0.3))
C = 1.0 / (exp_theta_adj + 1)
nCeta = 35 * C * exp_theta_adj
v1 = np.diagonal(neg_precQ, axis1=1, axis2=2) - nCeta
v2 = neg_precQ[:, None, 0, 0] - nCeta
v1, v2
```

```python
grad = (
    # dotJI_vmap(neg_precQ_a, neg_precQ_b, theta_m0)
    +jnp.matmul(neg_precQ[None], theta_m0[:, :, :, None])[..., 0]
    + y[:, None]
    - nCeta
)
hess = neg_precQ[None] - ((nCeta * C)[:, :, None, :] * jnp.eye(n_arms))
hess_a = neg_precQ_a[None, :, None] - nCeta * C
hess_b = neg_precQ_b
hess_a = (jnp.diagonal(neg_precQ, axis1=1, axis2=2) - nCeta * C) - hess_b[None, :, None]
```

```python

```

![](2022-07-07-20-14-45.png)

```python
import numpy as np

b = 1.0
a = np.random.rand(4)
M = np.full((4, 4), b) + np.diag(a)
```

```python
Minv = np.linalg.inv(M)
Minv
```

```python
b = neg_precQ[0, 0, 1]
a = neg_precQ[0, 0, 0] - b
```

```python
@jax.jit
def quad(theta_max, a, b):
    dotprod = (theta_max.sum(axis=-1) * b)[..., None] + theta_max * a
    quad = jnp.sum(theta_max * dotprod, axis=-1)
    return quad
```

```python
%%timeit
quad(theta_max, a, b).block_until_ready()
```

```python
quad(theta_max, a, b)[0, 0]
```

```python
dotprod2 = jnp.einsum("...ij,...j", neg_precQ, theta_max)
quad2 = jnp.einsum("...i,...ij,...j", theta_max, neg_precQ, theta_max)
quad3 = np.sum(theta_max * dotprod2, axis=-1)
```

```python
dotprod[0, 0], dotprod2[0, 0], quad[0, 0], quad2[0, 0], quad3[0, 0]
```

```python
inla_obj = inla.INLA(conditional_vmap, grad_hess_vmap, sig2_rule, narms)
```

```python
theta_max, hess, iters = inla_obj.optimize_loop(data, sig2_rule.pts, 1e-3)
post = inla_obj.posterior(theta_max, hess, sig2_rule.pts, sig2_rule.wts, data)
```

```python
%%timeit -n 20 -r 5
theta_max, hess, iters = inla_obj.optimize_loop(data, sig2_rule.pts, 1e-3)
post = inla_obj.posterior(theta_max, hess, sig2_rule.pts, sig2_rule.wts, data)
```

```python

```
