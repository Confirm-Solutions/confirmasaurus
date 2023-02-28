```python
import imprint.nb_util as nb_util
nb_util.setup_nb()

import time
import jax
import jax.numpy as jnp
from jax.scipy.special import expit, logit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import confirm.models.wd41 as wd41
import imprint as ip
```

## Exploring

We have two subgroups, each split equally into treatment and control arms:
- $p_{\mathrm{TNBC}}^{c}$ - TNBC subgroup control arm effectiveness.
- $p_{\mathrm{TNBC}}^{t}$ - TNBC subgroup treatment arm effectiveness.
- $p_{\mathrm{HR+}}^{c}$ - HR+ subgroup control arm effectiveness.
- $p_{\mathrm{HR+}}^{t}$ - HR+ subgroup treatment arm effectiveness.
  
$f_{\mathrm{TNBC}}$ is the fraction of patients in the TNBC subgroup.

The null hypotheses here are:

$$
p_{\mathrm{TNBC}}^{c} > p_{\mathrm{TNBC}}^{t}
$$

$$
f_{\mathrm{TNBC}} p_{\mathrm{TNBC}}^{c} + (1 - f_{\mathrm{TNBC}}) p_{\mathrm{HR+}}^{c} > 
f_{\mathrm{TNBC}} p_{\mathrm{TNBC}}^{t} + (1 - f_{\mathrm{TNBC}}) p_{\mathrm{HR+}}^{t}
$$


Adaptive Dunnett tests for treatment selection: https://pubmed.ncbi.nlm.nih.gov/17876763/


## Bigger job

```python
from confirm.adagrid import ada_calibrate
```

```python
model = wd41.WD41(0, 1, ignore_intersection=True)
grid = ip.cartesian_grid(
    [-2.5, -2.5, -2.5, -2.5],
    [1.0, 1.0, 1.0, 1.0],
    n=[10, 10, 10, 10],
    null_hypos = model.null_hypos
)
```

```python
db = ada_calibrate(
    wd41.WD41,
    g=grid,
    alpha=0.025,
    bias_target=0.005,
    grid_target=0.005,
    std_target=0.01,
    n_K_double=6,
    calibration_min_idx=80,
    step_size=2**13,
    packet_size=2**12,
    model_kwargs={'ignore_intersection': True}
)
```

```python
results_df = db.get_results()
```

```python
results_df['impossible']
```

```python
db.next(8192, "orderer")
```

```python
g_r = ip.grid.Grid(db.get_results(), None).prune_inactive()
```

```python
plt.plot(np.sort(g_r.df['orderer']))
plt.show()
```

```python
g_r.n_tiles
```

```python
B_lams = np.array([g_r.df[f'B_lams{i}'].min() for i in range(50)])
lamss = g_r.df['lams'].min()
B_lams, lamss, B_lams.mean(), lamss - B_lams.mean()
```

```python

plt.hist(B_lams)
plt.axvline(lamss, color='r')
plt.show()
```

```python
ordering = g_r.df['orderer'].sort_values()
ordering
```

```python
plt.plot(ordering.values)
plt.ylim([0.014, 0.018])
plt.show()
```

```python
worst_tile = g_r.df.loc[g_r.df['lams'].idxmin()]
worst_tile[['theta0', 'theta1', 'radii0', 'radii1', 'orderer', "alpha0", 'K', 'lams']]
```

```python
np.searchsorted(ordering, worst_tile['orderer']), ordering.shape
```

```python
B_worst_tile = [g_r.df.loc[g_r.df[f'B_lams{i}'].idxmin()] for i in range(50)]
[(B_worst_tile[i]['orderer'], np.searchsorted(ordering, B_worst_tile[i]['orderer'])) for i in range(50)]
```

```python
theta = g_r.get_theta()
p = expit(theta)
f = model.true_frac_tnbc
ptt = expit(np.linspace(-2.5, theta[:,1].max(), 100))
pht = ((p_tnbc_c * f + p_hrplus_c * (1 - f)) - (ptt * f)) / (1 - f)
plt.scatter(theta[:,0], theta[:,1], c=g_r.df['lams'], s=2, vmin=0.013, vmax=0.03)
plt.axvline(theta_tnbc_c, color='red', linewidth=4)
plt.plot(logit(ptt), logit(pht), color='red', linewidth=4)
cbar = plt.colorbar()
cbar.set_label('$\lambda^{*}$')
plt.xlabel(r'$\theta_{TNBC}^{T}$')
plt.ylabel(r'$\theta_{HR+}^{T}$')
plt.xlim([-2.5, 0.0])
plt.ylim([-2.5, 0.0])
plt.title('$\lambda^{**}='+f'{lamss:.4f}$')
plt.show()
```

```python
from jax.scipy.special import expit
results = jax.vmap(model.sim, in_axes=(0, None, None, None, None, None))(
    model.unifs[:1000], p_tnbc_c, expit(-1.0), p_hrplus_c, expit(1.0), True
)

lamss = 0.0239
import pandas as pd
df = pd.DataFrame(results)
df['rej_full'] = df['full_stat'] < lamss
df['rej_tnbc'] = df['tnbc_stat'] < lamss
df.head()
```

```python
df.loc[df['rej_tnbc']]
```
