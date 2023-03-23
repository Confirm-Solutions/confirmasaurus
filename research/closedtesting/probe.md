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
```

```python

ip.setup_nb()

db = ada.DuckDBTiles.connect('../../wd41_4d_v47')
```

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
db.con.query('select K, count(*) as n_tiles from tiles where active=true group by K order by K').df()
```

```python
lams = db.con.query('select lams from results where active=true').df()
lamss = lams['lams'].min()
max_display = lamss * 2
plt.hist(lams['lams'], bins=np.linspace(lamss, max_display, 100))
plt.show()
```

```python
report_df = db.get_reports()
working_reports = report_df[report_df['status'] == 'WORKING']
new_step_reports = report_df[report_df['status'] == 'NEW_STEP']
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
new_step_runtime = new_step_reports['runtime_total'].sum()
new_step_runtime / total_runtime
plt.plot(new_step_reports['runtime_total'], new_step_reports)
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
plot_df.shape
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

```python
reports_df = ch.query_df(client, 'select * from reports')
```
