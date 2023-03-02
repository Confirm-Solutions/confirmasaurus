```python
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from scipy.stats import norm
```

```python
B = 10
n_grid = 10
```

```python
def moments(k, mu=0, sig=1):
    m = (sig * np.sqrt(2)) ** k / np.sqrt(np.pi)
    m *= scipy.special.gamma(0.5 * (1 + k))
    m *= scipy.special.hyp1f1(-0.5 * k, 0.5, -0.5 * (mu / sig) ** 2)
    return m
```

```python
xs = np.linspace(0, B, num=n_grid)
ms = [moments(k) for k in range(n_grid)]
axs = np.abs(xs)
M = np.array([
    axs ** k
    for k in range(n_grid)
])
ws = np.linalg.solve(M, ms)
ws
```

The weights can be outside $[0,1]$ range it seems...
