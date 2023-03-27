```python
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from scipy.stats import norm
import cvxpy as cp
import matplotlib.pyplot as plt
```

```python
B = 50
n_grid = 1000
n_moments = 7
t = 2
```

```python
# E[|X|^k], X ~ N(mu, sig)
def moments(k, mu=0, sig=1):
    m = (sig * np.sqrt(2)) ** k / np.sqrt(np.pi)
    m *= scipy.special.gamma(0.5 * (1 + k))
    m *= scipy.special.hyp1f1(-0.5 * k, 0.5, -0.5 * (mu / sig) ** 2)
    return m
```

```python
xs = np.linspace(0, B, num=n_grid)
ms = [moments(k) for k in range(1, n_moments+1)]
M = np.array([
    xs ** k
    for k in range(1, n_moments+1)
])
```

```python
w = cp.Variable(n_grid)
objective = cp.Maximize(
    cp.sum(w[xs > t])
)
constraints = [
    0 <= w,
    w <= 1,
    cp.sum(w) == 1,
    M @ w == ms,
]
prob = cp.Problem(objective, constraints)
result = prob.solve(
    solver='ECOS',
)
```

```python
print(result) # P(|X| > t) under F^*
print( 
    (M @ w.value) / np.array([t ** i for i in range(1, n_moments+1)])
)
print(2 * scipy.stats.norm.sf(t)) # P(|X| > t)
```
