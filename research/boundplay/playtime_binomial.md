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
import confirm.outlaw.nb_util as nb_util

nb_util.setup_nb()
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from numpy import sqrt
import jax
import jax.numpy as jnp
```

```python
n = 50
p = 0.2
t = jax.scipy.special.logit(p)
At = lambda t: -n * jnp.log(1 - jax.scipy.special.expit(t))
g = lambda t, x: t * x - At(t)
dg = jax.grad(g)
dg_vmap = jax.vmap(dg, in_axes=(None, 0))

holderp = 1.2
holderq = 1.0 / (1 - 1.0 / holderp)

# numerical integral to compute E[ |grad g|^q]^(1/q)
xs = jnp.arange(n + 1).astype(jnp.float64)
eggq = jnp.abs(dg_vmap(t, xs)) ** holderq
C = sum(eggq * scipy.stats.binom.pmf(xs, n, p)) ** (1 / holderq)
C
```

```python
f0 = 0.001
dt = 0.5
t_path = np.linspace(t, t + dt, 100)
def derivs(t, y):
    cur_f = y[0]
    return C * cur_f ** (1 / holderp)
solution = scipy.integrate.solve_ivp(derivs, (t, t+dt), [f0], t_eval=t_path)
```

```python
grad_bound = derivs(t, [f0])
hess_bound = n * p * (1 - p)
classical = f0 + grad_bound * (t_path - t) + 0.5 * hess_bound * (t_path - t) ** 2
```

```python
plt.plot([t], [f0], 'ko')
plt.plot(t_path, solution['y'][0,:], 'b-')
plt.plot(t_path, classical, 'r--')
plt.xlabel(r'$\theta$')
plt.ylabel('type I error')
plt.show()
```

```python
idx = np.argmin(solution['y'][0, :] < 0.025)
```

```python
t_path[idx]
```

```python
classical[-1] / solution['y'][0,-1]
```

```python
solution['y'][0,-1]
```
