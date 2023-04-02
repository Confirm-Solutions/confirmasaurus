```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import jax

import imprint as ip
import confirm
import confirm.adagrid as ada
from confirm.adagrid.const import MAX_STEP
import confirm.cloud.clickhouse as ch

ip.setup_nb()

db = 'wd41_4d_v74'
ddb_path = Path(confirm.__file__).parent.parent.joinpath(f'{db}.db')

ch_client = ch.connect(db, service='PROD')
def query(q):
    # return ddb.con.query(q).df()
    return ch.query_df(ch_client, q)
ddb = ada.DuckDBTiles.connect(str(ddb_path))
```

```python
dim = len([c for c in query('select * from results limit 1').columns if c.startswith('theta')])
n_steps = query('select max(step_id) from results').iloc[0][0] + 1
dim, n_steps
```

## Broad table exploration

```python
n_rows_df = pd.DataFrame(
    [(table, query(f'select count(*) from {table}').iloc[0][0]) for table in ch.all_tables],
    columns=['table', 'count']
).set_index('table')
n_rows_df
```

```python
from confirm.adagrid.const import MAX_STEP
n_sims = query('select sum(K) from results').iloc[0][0]
n_retained_sims = query(f'select sum(K) from results where inactivation_step={MAX_STEP} and id not in (select id from done where active=false)').iloc[0][0]
n_sims / 1e12, n_retained_sims / 1e12
```

```python
n_active_tiles = query(
    f"select count(*) from results where inactivation_step={MAX_STEP}"
).iloc[0][0]
n_eligible_tiles = query(
    f"select count(*) from results where completion_step={MAX_STEP}"
).iloc[0][0]
n_active_tiles, n_eligible_tiles
```

```python
query(f'select K, count(*) as n_tiles from tiles where inactivation_step={MAX_STEP} group by K order by K')
```

```python
volume_sql = (
    str(2 ** dim)
    + "*"
    + ("*".join([f"radii{d}" for d in range(dim)]))
)
```

```python
smallest_tile = query(f'select * from results where inactivation_step={MAX_STEP} order by {volume_sql} limit 1')
smallest_tile
```

```python
largest_tile = query(f'select * from results where inactivation_step={MAX_STEP} order by {volume_sql} desc limit 1')
largest_tile
```

```python
n_possible_tiles = query('select count(*) from results where isNotNull(lams)').iloc[0][0]
n_possible_tiles
```

```python
lamss_tile = query(f'''
select * from results 
    where inactivation_step={MAX_STEP} 
        and isNotNull(lams) 
    order by lams 
    limit 1
''')
lamss_tile
```

```python
lamss = lamss_tile['lams'].iloc[0]
```

```python
lams = query(f'''
select lams from results 
    where inactivation_step={MAX_STEP} 
        and lams <= {lamss}
''')
max_display = lamss * 2
plt.hist(lams['lams'], bins=np.linspace(lamss, max_display, 100))
plt.show()
```

## Ordering

```python
worst5000_df = self.con.query('select * from results where inactivation_step>10000 order by lams limit 5000').df()
```

```python
orderer_df = self.con.query(
    f"select orderer from results where inactivation_step>10000 and orderer <= {worst5000_df['orderer'].max()} order by orderer"
).df()

```

```python
wait = np.searchsorted(orderer_df['orderer'], worst5000_df['orderer'])
wait
```

```python
cfg = self.get_config().iloc[0].to_dict()
```

```python
import confirm.adagrid.calibrate as adacal
import confirm.models.wd41 as wd41
report = dict()
algo = adacal.AdaCalibrate(wd41.WD41(0, 600000, ignore_intersection=True), None, self, cfg, None)

```

```python
tiles_df = self.next(10, 11, 50000, 'orderer')
twb_worst_tile = self.worst_tile(10, "twb_mean_lams")
# np.sum(tiles_df['twb_min_lams'] > twb_worst_tile['twb_mean_lams'].iloc[0])
```

```python
self = algo
twb_worst_tile = self.db.worst_tile(10, "twb_mean_lams")
for col in twb_worst_tile.columns:
    if col.startswith("radii"):
        twb_worst_tile[col] = 1e-6
twb_worst_tile["K"] = self.max_K
twb_worst_tile_lams = self.driver.bootstrap_calibrate(
    twb_worst_tile,
    self.cfg["alpha"],
    calibration_min_idx=self.cfg["calibration_min_idx"],
    tile_batch_size=1,
)
twb_worst_tile_mean_lams = twb_worst_tile_lams["twb_mean_lams"].iloc[0]
deepen_likely_to_work = tiles_df["twb_min_lams"] > twb_worst_tile_mean_lams
```

```python
np.sum(deepen_likely_to_work)
```

## Investigating the WD41 problem.

```python
import confirm.models.wd41 as wd41

lamss_potential = lamss_tile.copy()
for d in range(self.dimension()):
    lamss_potential[f"radii{d}"] = 1e-7
cal_df = ip.calibrate(
    wd41.WD41,
    g=ip.Grid(lamss_potential, 1),
    model_kwargs={"ignore_intersection": True, "dtype": np.float64},
)
```

```python
debug_df = pd.concat(
    (
        lamss_potential[
            [f"theta{d}" for d in range(self.dimension())]
            + [f"radii{d}" for d in range(self.dimension())]
        ],
        cal_df,
    ),
    axis=1,
)
for d in range(self.dimension()):
    name = ['p_{hr+, c}', 'p_{hr+, t}', 'p_{tnbc, c}', 'p_{tnbc, t}'][d]
    debug_df[name] = scipy.special.expit(debug_df[f"theta{d}"])
debug_df.drop(columns=[f"theta{d}" for d in range(self.dimension())], inplace=True)
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
report_df = self.get_reports()
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
working_reports
```

```python
working_reports['runtime_simulating'].sum()
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
import numpy as np
xs = np.linspace(-2, 1, 10)
ys = np.linspace(-2, 1, 10)
counts = np.empty((len(xs), len(ys)))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        NN = self.con.query(f'''
            select count(*) 
                from results
                where 
                    inactivation_step={MAX_STEP}
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
plot_df = self.con.query('''
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