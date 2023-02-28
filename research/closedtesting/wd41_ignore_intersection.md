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

```python
m = wd41.WD41(0, 1, ignore_intersection=True)
```

```python
p_tnbc_c = 0.34
p_hrplus_c = 0.23
S = np.linspace(0, 0.3, 7)
Sbar = np.linspace(0, 0.2, 3)
SS = np.stack(np.meshgrid(S, Sbar, indexing='ij'), axis=-1).reshape(-1, 2)
t_tnbc_t = logit(p_tnbc_c + SS[:, 0])
t_tnbc_c = np.full_like(t_tnbc_t, logit(p_tnbc_c))
t_hrplus_t = logit(p_hrplus_c + SS[:, 1])
t_hrplus_c = np.full_like(t_hrplus_t, logit(p_hrplus_c))
theta = np.stack((t_tnbc_c, t_tnbc_t, t_hrplus_c, t_hrplus_t), axis=-1)
```

```python
K = 300000
np.random.seed(0)
unifs = np.random.uniform(size=(K, m.unifs.shape[1]))
info = m.sim_jit(unifs, theta, True)
```

```python
lam = 0.014290583
df = pd.DataFrame(SS, columns=['Effect S', 'Effect Sbar'])
df['Effect F'] = 0.54 * df['Effect S'] + 0.46 * df['Effect Sbar']
df['Power'] = np.sum((info['tnbc_stat'] < lam) | (info['full_stat'] < lam), axis=1) / K
df['P(Reject F)'] = np.sum(info['full_stat'] < lam, axis=1) / K
df['P(Reject S)'] = np.sum(info['tnbc_stat'] < lam, axis=1) / K
df['P(Select F)'] = np.sum(info['hypofull_live'], axis=1) / K
df['P(Select S)'] = np.sum(info['hypotnbc_live'], axis=1) / K
df['P(Select 1 set)'] = np.sum((~info['hypotnbc_live']) | (~info['hypofull_live']), axis=1) / K
df['P(Enrichment)'] = 1 - df['P(Select F)']
df.set_index(np.arange(1, 22), inplace=True)
df
```

```python
df_compare = pd.read_csv('wd41.csv', index_col=0)
```

```python
df['Power'] - df_compare['Power']
```

```python
theta0 = np.linspace(-3, 3, 100)
theta1 = theta0
theta2 = np.full_like(theta0, -1)
theta3 = np.full_like(theta0, 10)
theta = np.stack((theta0, theta1, theta2, theta3), axis=-1)
nt0 = np.ones_like(theta0)
nt1 = np.zeros_like(theta0)
null_truth = np.stack((nt0, nt1), axis=-1)

tie_sum = np.zeros_like(theta0)
nsims = 0
for i in range(50):
    unifs = np.random.random(size=(100000, m.unifs.shape[1]))
    stats, _ = m.sim_jit(unifs, theta, False)
    tie_sum += (stats < lam).sum(axis=1)
    nsims += unifs.shape[0]
    print(nsims)

tie_est = tie_sum / nsims
plt.plot(theta0, 100 * tie_est)
plt.xlabel(r'$\theta_{TNBC}^c = \theta_{TNBC}^t$')
plt.ylabel('Type I Error (\%)')
plt.title(r'Sharp Null for TNBC, Deep alternative for HR+')
plt.suptitle(r'$\theta_{HR+}^c = -1, \theta_{HR+}^t = 10, \lambda = 0.025$')
plt.show()
```

## Simple grids

```python
theta_tnbc_c = -1
p_tnbc_c = expit(theta_tnbc_c)
theta_hrplus_c = -1
p_hrplus_c = expit(theta_hrplus_c)


def get_theta(theta):
    t_tnbc_t = theta[..., 0]
    return jnp.stack(
        (
            jnp.full_like(t_tnbc_t, theta_tnbc_c),
            t_tnbc_t,
            jnp.full_like(t_tnbc_t, theta_hrplus_c),
            theta[..., 1],
        ),
        axis=-1,
    )


class WD41Null2D(wd41.WD41Null):
    def get_theta(self, theta):
        return get_theta(theta)


class WD412D(wd41.WD41):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.family_params = {
            "n": jnp.array(
                [
                    self.n_max_tnbc,
                    self.n_max_hrplus,
                ]
            )
        }

    def sim_batch(
        self,
        begin_sim: int,
        end_sim: int,
        theta: jnp.ndarray,
        null_truth: jnp.ndarray,
        detailed: bool = False,
    ):
        return super().sim_batch(
            begin_sim,
            end_sim,
            get_theta(theta),
            null_truth,
            detailed,
        )
```

```python
model = WD412D(0, 100000, ignore_intersection=True)
nulls = [ip.hypo("-1 > theta0"), WD41Null2D(model.true_frac_tnbc)]
```

```python
grid = ip.cartesian_grid([-2.5, -2.5], [-0.0, -0.0], n=[90, 90], null_hypos=nulls)

val_df = ip.validate(WD412D, g=grid, lam=lam, K=10000, model_kwargs={'ignore_intersection': True})
```

```python
theta = grid.get_theta()
p = expit(theta)
f = model.true_frac_tnbc
ptt = expit(np.linspace(-2.5, theta[:,1].max(), 100))
pht = ((p_tnbc_c * f + p_hrplus_c * (1 - f)) - (ptt * f)) / (1 - f)
```

```python
plt.scatter(theta[:,0], theta[:,1], c=val_df['tie_est'], s=8)
plt.axvline(theta_tnbc_c, color='red', linewidth=4)
plt.plot(logit(ptt), logit(pht), color='red', linewidth=4)
cbar = plt.colorbar()
cbar.set_label('$\hat{f}$')
plt.xlabel(r'$\theta_{TNBC}^{T}$')
plt.ylabel(r'$\theta_{HR+}^{T}$')
plt.xlim([-2.5, 0.0])
plt.ylim([-2.5, 0.0])
plt.show()
```

```python
cal_df = ip.calibrate(WD412D, g=grid, alpha=0.025, K=10000, model_kwargs={'ignore_intersection': True})
```

```python
plt.scatter(theta[:,0], theta[:,1], c=cal_df['lams'], s=8, vmin=0.02, vmax=0.05)
plt.axvline(theta_tnbc_c, color='red', linewidth=4)
plt.plot(logit(ptt), logit(pht), color='red', linewidth=4)
cbar = plt.colorbar()
cbar.set_label('$\lambda^{*}$')
plt.xlabel(r'$\theta_{TNBC}^{T}$')
plt.ylabel(r'$\theta_{HR+}^{T}$')
plt.xlim([-2.5, 0.0])
plt.ylim([-2.5, 0.0])
plt.title('$\lambda^{**}='+f'{cal_df["lams"].min():.4f}$')
plt.show()
```

## Bigger job

```python
from confirm.adagrid import ada_calibrate
```

```python
grid = ip.cartesian_grid(
    [-2.5, -2.5],
    [1.0, 1.0],
    n=[10, 10],
    null_hypos = nulls
)
```

```python
db = ada_calibrate(
    WD412D,
    g=grid,
    alpha=0.025,
    bias_target=0.0005,
    grid_target=0.0005,
    std_target=0.0015,
    n_K_double=6,
    calibration_min_idx=80,
    step_size=2**13,
    packet_size=2**12,
    model_kwargs={'ignore_intersection': True}
)
```

```python
g_r = ip.grid.Grid(db.get_results(), None).prune_inactive()
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
