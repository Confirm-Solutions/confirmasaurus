```python
import sys

sys.path.append("../imprint/research/berry/")
import berrylib.util as util

util.setup_nb()
```

```python
import numpy as np
import scipy.stats
import scipy.spatial
import matplotlib.pyplot as plt
import berrylib.fast_inla as fast_inla

mu_mean = 0.1856
mu_sig2 = 0.05
sig2_alpha = 5.0
sig2_beta = 0.3
alpha = [24, 66, 40]
beta = [416.3, 331, 234]
lambdaj = scipy.stats.invgamma.mean(alpha, scale=beta)

fi = fast_inla.FastINLASurvival(
    lambdaj=lambdaj,
    n_arms=3,
    mu_0=mu_mean,
    mu_sig2=mu_sig2,
    sigma2_n=20,
    sigma2_bounds=[0.006401632420120484, 2.8421994410275007],
    sigma2_alpha=sig2_alpha,
    sigma2_beta=sig2_beta,
)
```

```python
data = np.array(
    [
        [
            [25.0, 490.89548468, 36.0],
            [39.0, 183.41089701, 39.0],
            [21.0, 107.54346258, 21.0],
        ]
    ]
)
data = np.tile(data, (3, 1, 1))


def test(data):
    sigma2_post, exceedance, theta_max, theta_sigma, _ = fi.numpy_inference(
        data, thresh_theta=np.repeat(np.log(1.0), 3)
    )
    interp = (exceedance[1] + exceedance[0]) * 0.5
    return exceedance[2] - interp
```

```python
for i in range(1, 8):
    exp = data.copy()
    exp[1, :, 0] += i
    exp[2, :, 0] += i * 0.5
    err = test(exp)
    print(f"stepsize={i} err={err}")
```

```python
for i in range(1, 40, 4):
    exp = data.copy()
    exp[1, :, 1] += i
    exp[2, :, 1] += i * 0.5
    err = test(exp)
    print(f"stepsize={i} err={err}")
```

```python
n_events = np.arange(40).astype(np.float64)
total_time = np.arange(0, 600, 20).astype(np.float64)
NE, TT = np.meshgrid(n_events, total_time, indexing="ij")
d0_pairs = np.stack((NE, TT), axis=-1).reshape((-1, 2))
d0_pairs
grid = np.tile(d0_pairs[:, None], (1, 3, 1))
```

```python
sigma2_post, exceedance, theta_max, theta_sigma, _ = fi.numpy_inference(
    grid, thresh_theta=np.repeat(np.log(1.0), 3)
)
```

```python
grid[400]
```

```python
exceedance[400]
```

```python
ex_grid = exceedance.reshape((*NE.shape, 3))
levels = None
levels = [0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0]
levels = np.linspace(0, 1, 31)
cntf = plt.contourf(NE, TT, ex_grid[:, :, 0], levels=levels, extend="both")
plt.contour(
    NE,
    TT,
    ex_grid[:, :, 0],
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.show()
```

```python
pt = np.tile(
    np.array([np.random.uniform(0, 40), np.random.uniform(0, 600)])[None, None],
    (1, 3, 1),
)
_, pt_exceedance, _, _, _ = fi.numpy_inference(
    pt, thresh_theta=np.repeat(np.log(1.0), 3)
)
scale = 1.0 / np.array([1.0, 20.0])[None, :]
kdtree = scipy.spatial.KDTree(grid[:, 0] * scale)
dist, idx = kdtree.query(pt[0, 0] * scale, k=3)
pt_exceedance, exceedance[idx]
```

```python
dist, pt, grid[idx]
```

```python
pt_exceedance, exceedance[idx[0, 0]], grid[idx[0, 0]]
```

```python
tri = grid[idx[0], 0, :] * scale
T = tri[:2] - tri[2]
l1, l2 = np.linalg.solve(T, (pt[0, 0] * scale[0] - tri[2]))
l3 = 1.0 - l1 - l2
l1, l2, l3
```

```python
interp_exceedance = np.sum(exceedance[idx[0]] * np.array([l1, l2, l3])[:, None], axis=0)
```

```python
pt_exceedance, interp_exceedance
```

```python
err = []
for i in range(1000):
    pt = np.tile(
        np.array([np.random.uniform(0, 40), np.random.uniform(0, 600)])[None, None],
        (1, 3, 1),
    )
    try:
        _, pt_exceedance, _, _, _ = fi.numpy_inference(
            pt, thresh_theta=np.repeat(np.log(1.0), 3)
        )
    except:
        continue

    scale = 1.0 / np.array([1.0, 20.0])[None, :]
    kdtree = scipy.spatial.KDTree(grid[:, 0] * scale)
    dist, idx = kdtree.query(pt[0, 0] * scale, k=3)

    tri = grid[idx[0], 0, :] * scale
    try:
        T = (tri[:2] - tri[2]).T
        l1, l2 = np.linalg.solve(T, (pt[0, 0] * scale[0] - tri[2]))
    except:
        continue
    l3 = 1.0 - l1 - l2

    interp_exceedance = np.sum(
        exceedance[idx[0]] * np.array([l1, l2, l3])[:, None], axis=0
    )
    pt[0, 0], tri / scale, exceedance[idx[0]],
    err.append(pt_exceedance - interp_exceedance)
    # print('')
    # print('true exceedance', pt_exceedance)
    # print('interpolated exceedance', interp_exceedance)
err = np.array(err)
```

```python
plt.hist(np.log10(np.abs(err[:, 0, 0])), bins=np.linspace(-6, -1, 21))
plt.show()
```

```python

```

```python
tri, pt[0, 0] * scale[0]
```

```python
v = pt[0, 0] * scale[0] - tri[2]
tri[:2] - tri[2], v
```

```python
soln = np.linalg.inv((tri[:2] - tri[2]).T).dot(v)
```

```python
soln
```

```python
tri[:2] - tri[2]
```

```python
l1, l2, l3
```

```python

```
