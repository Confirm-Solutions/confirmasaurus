```python
import confirm.cloud.clickhouse as ch
# client = ch.connect('wd41_4d_v0')
# ch.list_tables(client)
```

```python
import confirm.adagrid as ada
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imprint as ip
import scipy.special
import jax
```

```python

ip.setup_nb()

db = ada.DuckDBTiles.connect('../../wd41_4d_v47')
```

## Broad table exploration

```python
n_rows_df = pd.DataFrame(
    [(table, db.con.query(f'select count(*) from {table}').fetchone()[0]) for table in ch.all_tables],
    columns=['table', 'count']
).set_index('table')
n_rows_df

```

```python
n_sims = db.con.query('select sum(K) from results').fetchone()[0]
n_retained_sims = db.con.query('select sum(K) from results where active=true').fetchone()[0]
n_sims / 1e12, n_retained_sims / 1e12
```

```python
n_active_tiles = db.con.query(
    "select count(*) from results where active=true"
).fetchone()[0]
n_eligible_tiles = db.con.query(
    "select count(*) from results where eligible=true"
).fetchone()[0]
n_active_tiles, n_eligible_tiles
```

```python
db.con.query('select K, count(*) as n_tiles from tiles where active=true group by K order by K').df()
```

```python
volume_sql = (
    str(2 ** db.dimension())
    + "*"
    + ("*".join([f"radii{d}" for d in range(db.dimension())]))
)

db.con.execute(f'create index if not exists volume_idx2 on results (active, ({volume_sql}))')
```

```python
smallest_tile = db.con.query(f'select * from results where active=true order by {volume_sql} limit 1').df()
smallest_tile
```

```python
largest_tile = db.con.query(f'select * from results where active=true order by {volume_sql} desc limit 1').df()
largest_tile
```

```python
lamss_tile = db.con.query('select * from results where active=true order by lams limit 1').df()
lamss_tile
```

```python
lams = db.con.query('select lams from results where active=true').df()
lamss = lams['lams'].min()
max_display = lamss * 2
plt.hist(lams['lams'], bins=np.linspace(lamss, max_display, 100))
plt.show()
```

## Investigating the WD41 problem.

```python
import confirm.models.wd41 as wd41

lamss_potential = lamss_tile.copy()
for d in range(db.dimension()):
    lamss_potential[f"radii{d}"] = 1e-7
cal_df = ip.calibrate(
    wd41.WD41,
    g=ip.Grid(lamss_potential, 1),
    model_kwargs={"ignore_intersection": True, "dtype": np.float64},
)
```

```python
lamss_potential
```

```python
debug_df = pd.concat(
    (
        lamss_potential[
            [f"theta{d}" for d in range(db.dimension())]
            + [f"radii{d}" for d in range(db.dimension())]
        ],
        cal_df,
    ),
    axis=1,
)
for d in range(db.dimension()):
    name = ['p_{hr+, c}', 'p_{hr+, t}', 'p_{tnbc, c}', 'p_{tnbc, t}'][d]
    debug_df[name] = scipy.special.expit(debug_df[f"theta{d}"])
debug_df.drop(columns=[f"theta{d}" for d in range(db.dimension())], inplace=True)
debug_df
```

```python
model = wd41.WD41(0, 10000, ignore_intersection=True)
g_explore = ip.cartesian_grid([-0.6, -1.5875, -0.6, 0.5125], [-0.4, -1.5875, -0.4, 0.5125], n=[10, 1, 10, 1], null_hypos=model.null_hypos)
cal_df = ip.calibrate(wd41.WD41, g=g_explore, model_kwargs={"ignore_intersection": True, "dtype": np.float64})
cal_df = pd.concat((g_explore.df, cal_df), axis=1)
```

```python
cal_df['lams'].min()
```

```python
plt.subplot(2,2,1)
plt.scatter(cal_df['theta0'], cal_df['theta2'], c=cal_df['null_truth0'])
plt.colorbar()
plt.subplot(2,2,2)
plt.scatter(cal_df['theta0'], cal_df['theta2'], c=cal_df['null_truth1'])
plt.colorbar()
plt.subplot(2,2,3)
plt.scatter(cal_df['theta0'], cal_df['theta2'], c=cal_df['lams'])
plt.colorbar()
plt.show()
```

```python
p = scipy.special.expit(g_explore.get_theta()[40])
sim_vmap = jax.vmap(model.sim, in_axes=(0, None, None, None, None, None))
stats = sim_vmap(model.unifs, *p, True)
# print(np.percentile(stats, 99))
# plt.plot(np.sort(stats[0]))
# plt.show()
```

```python
bad_idx = stats['full_stat'].argmin()
model.sim(model.unifs[bad_idx], *p, True)
```

```python
(
    phattnbccontrol,
    phattnbctreat,
    phathrpluscontrol,
    phathrplustreat,
) = model.sample_all_arms(model.unifs[bad_idx][model.n_first_stage :], *p)

```

```python
import jax.numpy as jnp
totally_pooledaverage = (
    (phattnbctreat + phattnbccontrol) * model.n_tnbc_first_stage_per_arm
    + (phathrplustreat + phathrpluscontrol) * model.n_hrplus_first_stage_per_arm
) / model.n_first_stage
denominatortotallypooled = jnp.sqrt(
    totally_pooledaverage
    * (1 - totally_pooledaverage)
    # * (1 / (model.n_first_stage))
    * ((1 / (2 * model.n_tnbc_first_stage_per_arm)) + (1 / (2 * model.n_hrplus_first_stage_per_arm)))
)
tnbc_effect = phattnbctreat - phattnbccontrol
hrplus_effect = phathrplustreat - phathrpluscontrol
zfull = (
    (tnbc_effect * model.n_tnbc_first_stage_total)
    + (hrplus_effect * model.n_hrplus_first_stage_total)
) / (denominatortotallypooled * model.n_first_stage)
```

```python
zfull
```

```python
((tnbc_effect * model.n_tnbc_first_stage_total)
+ (hrplus_effect * model.n_hrplus_first_stage_total)) / model.n_first_stage
```

```python
denominatortotallypooled
```

```python
phattnbccontrol, phattnbctreat, phathrpluscontrol, phathrplustreat
```

```python
def sample(unifs, p):
    return jnp.sum(unifs < p, dtype=model.dtype)

def sample_all_arms(
    self, unifs, pcontrol_tnbc, ptreat_tnbc, pcontrol_hrplus, ptreat_hrplus
):
    unifs_tnbc = unifs[: self.n_tnbc_second_stage_total]
    unifs_tnbc_control = unifs_tnbc[: self.n_tnbc_second_stage_per_arm]
    unifs_tnbc_treat = unifs_tnbc[self.n_tnbc_second_stage_per_arm :]
    unifs_hrplus = unifs[self.n_tnbc_second_stage_total :]
    unifs_hrplus_control = unifs_hrplus[: self.n_hrplus_second_stage_per_arm]
    unifs_hrplus_treat = unifs_hrplus[self.n_hrplus_second_stage_per_arm :]

    ytnbccontrol = sample(unifs_tnbc_control, pcontrol_tnbc)
    ytnbctreat = sample(unifs_tnbc_treat, ptreat_tnbc)
    yhrpluscontrol = sample(unifs_hrplus_control, pcontrol_hrplus)
    yhrplustreat = sample(unifs_hrplus_treat, ptreat_hrplus)
    return ytnbccontrol, ytnbctreat, yhrpluscontrol, yhrplustreat
```

```python
(
    ytnbccontrol,
    ytnbctreat,
    yhrpluscontrol,
    yhrplustreat,
) = sample_all_arms(model, model.unifs[bad_idx][model.n_first_stage :], *p)
```

```python
(
    ytnbccontrol,
    ytnbctreat,
    yhrpluscontrol,
    yhrplustreat,
)
```

```python
ycontrol = ytnbccontrol + yhrpluscontrol
ytreat = ytnbctreat + yhrplustreat
ncontrol = ntreat = (
    model.n_hrplus_first_stage_per_arm + model.n_tnbc_first_stage_per_arm
)
phattreat = ytreat / ntreat
phatcontrol = ycontrol / ncontrol
full_pooledaverage = (phattreat + phatcontrol) * 0.5
denominatorfull = jnp.sqrt(
    full_pooledaverage * (1 - full_pooledaverage) * (2 / ncontrol)
)
zfull = (phattreat - phatcontrol) / denominatorfull
zfull
```

```python

```

```python
print(np.percentile(stats['tnbc_stat'], 1))
print(np.percentile(stats['full_stat'], 1))
plt.subplot(1,2,1)
plt.plot(np.sort(stats['tnbc_stat']))
plt.subplot(1,2,1)
plt.plot(np.sort(stats['full_stat']))
plt.show()
```

## Looking at the reports

```python
report_df = db.get_reports()
working_reports = report_df[report_df['status'] == 'WORKING'].dropna(axis=1, how='all')
new_step_reports = report_df[report_df['status'] == 'NEW_STEP'].dropna(axis=1, how='all')
```

```python
sim_runtime = working_reports['runtime_simulating'].sum()
total_runtime = report_df.iloc[-1]['time'] - report_df.iloc[0]['time']
parallelism = sim_runtime / total_runtime
parallelism, sim_runtime / 3600, total_runtime / 3600
```

```python
plt.plot(working_reports['runtime_per_sim_ns'])
plt.ylim([0, np.percentile(working_reports['runtime_per_sim_ns'], 99.5)])
plt.show()
```

```python
max_working_time = working_reports.groupby('step_id')['time'].max()
total_sim_time = working_reports.groupby('step_id')['runtime_simulating'].sum()
ex_df = new_step_reports[['step_id', 'runtime_total', 'time']].set_index('step_id')
ex_df['max_working_time'] = max_working_time
ex_df['total_sim_time'] = total_sim_time
step_parallelism = ex_df['total_sim_time'] / ((ex_df['max_working_time'] - ex_df['time']) + ex_df['runtime_total'])
plt.plot(step_parallelism, 'k-o', markersize=3)
plt.xlabel('step_id')
plt.ylabel('parallelism')
plt.show()
```

```python
plt.plot(new_step_reports[['step_id', 'lamss']].set_index('step_id'))
plt.xlabel('step_id')
plt.ylabel('$\lambda^{**}$')
plt.show()
```

```python
plt.plot(new_step_reports['tie_{k}(lamss)'])
plt.show()
```

```python
plot_df = db.con.query('''
    select theta0, theta1, theta2, theta3, 
            radii0, radii1, radii2, radii3, 
            alpha0, K, lams, twb_mean_lams, twb_min_lams 
        from results
        where 
            abs(theta0 + 1) < 0.05
            and abs(theta2 + 1) < 0.05
''').df()
```

```python
import numpy as np
xs = np.linspace(-2, 1, 10)
ys = np.linspace(-2, 1, 10)
counts = np.empty((len(xs), len(ys)))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        NN = db.con.query(f'''
            select count(*) 
                from results
                where 
                    active=true
                    and abs(theta0 - {x}) < 0.167
                    and abs(theta2 - {y}) < 0.167
        ''').fetchone()[0]
        counts[i, j] = NN
```

```python
counts.sum() / 1e6
```

```python
XX, YY = np.meshgrid(xs, ys, indexing='ij')
# plt.scatter(XX.ravel(), YY.ravel(), c=counts.ravel())
plt.contourf(XX, YY, counts, levels=20)
plt.xlabel('$\\theta_{TNBC, c}$')
plt.ylabel('$\\theta_{HR+, c}$')
cbar = plt.colorbar()
cbar.set_label('$N$')
plt.show()
```

```python
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(plot_df['theta1'], plot_df['theta3'], c=plot_df['K'], s=5)
plt.colorbar()
plt.subplot(2, 2, 2)
plt.scatter(plot_df['theta1'], plot_df['theta3'], c=plot_df['alpha0'], s=5)
plt.colorbar()
plt.subplot(2, 2, 3)
plt.scatter(plot_df['theta1'], plot_df['theta3'], c=plot_df['lams'], s=5)
plt.colorbar()
plt.subplot(2, 2, 4)
plt.scatter(plot_df['theta1'], plot_df['theta3'], c=plot_df['twb_min_lams'], s=5)
plt.colorbar()
plt.show()
```
