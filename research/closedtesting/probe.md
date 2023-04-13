```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import jax
import duckdb

import imprint as ip
import confirm
import confirm.adagrid as ada
from confirm.adagrid.const import MAX_STEP
import confirm.cloud.clickhouse as ch

ip.setup_nb()

job_name_prefix = "wd41"
service = "PROD"

ddb_path = Path(confirm.__file__).parent.parent.joinpath(f"{job_name_prefix}.db")
use_clickhouse = False
if ddb_path.exists():
    try:
        ddb = ada.DuckDBTiles(duckdb.connect(str(ddb_path), read_only=True))

        def query(q):
            return ddb.con.query(q).df()

    except Exception:
        use_clickhouse = True
else:
    use_clickhouse = True

if use_clickhouse:
    client = ch.get_ch_client(service=service)
    jobs = [job_name for job_name in ch.list_dbs(client) if job_name.startswith(job_name_prefix)]
    jobs.sort()
    job_name = jobs[-1]

    ch_db = ch.ClickhouseTiles.connect(job_name, service=service)
    query = ch_db.query
```

```python
query('select count(*)/1000000 from results')
```

```python
query('select sum(K)/1e12 from results')
```

```python
query('select median(K) from results')
```

```python
dim = len(
    [c for c in query("select * from results limit 1").columns if c.startswith("theta")]
)
n_steps = query("select max(step_id) from results").iloc[0][0] + 1
dim, n_steps

```

```python
for m in query("select * from logs where message LIKE '%Command took%'")['message'].values:
    print(m)
```

```python
for m in query("select * from logs where message LIKE '%Query took%'")['message'].values:
    print(m)
```

## Performance

```python
import json

reports = [json.loads(v) for v in query("select * from reports")["json"].values]
report_df = pd.DataFrame(reports)
working_reports = report_df[report_df["status"] == "WORKING"].dropna(axis=1, how="all")
new_step_reports = report_df[(report_df["status"] == "NEW_STEP") | (report_df['status'] == 'CONVERGED')].dropna(
    axis=1, how="all"
)
new_step_reports.set_index("step_id", inplace=True)
new_step_reports.sort_index(inplace=True)

```

```python
min_runtime_per_sim_ns = working_reports["runtime_per_sim_ns"].min()
min_runtime_per_sim_ns = 25
```

```python
from matplotlib.gridspec import GridSpec

max_step = working_reports['step_id'].max()
min_step = max_step - 8
color_list = ["k", "r", "b", "m"]
offset = report_df["start_time"].min()

step_id = working_reports["step_id"]
include = (step_id >= min_step) & (step_id <= max_step)
df = working_reports[include][
    ['n_total_sims', "sim_start_time", "sim_done_time", "start_time", "done_time", "step_id", 'runtime_per_sim_ns']
].copy()
df.sort_values(by=["sim_start_time"], inplace=True)
df["adjusted_start_time"] = df["start_time"] - offset
df["adjusted_done_time"] = df["done_time"] - offset
df["adjusted_sim_start_time"] = df["sim_start_time"] - offset
df["adjusted_sim_done_time"] = df["sim_done_time"] - offset
min_time = df["adjusted_start_time"].min() - 10
max_time = df["adjusted_done_time"].max() + 10
df['sims_per_sec'] = 1e9 / df['runtime_per_sim_ns']

df["positions"] = (df.groupby("step_id").cumcount()) / 10.0
df["packet_linelengths"] = df["done_time"] - df["start_time"]
df["packet_lineoffsets"] = df["adjusted_start_time"] + df["packet_linelengths"] * 0.5
df["packet_linewidths"] = 1

df["positions"] = (df.groupby("step_id").cumcount()) / 10.0
df["linelengths"] = df["sim_done_time"] - df["sim_start_time"]
df["lineoffsets"] = (
    df["sim_start_time"] - report_df["start_time"].min() + df["linelengths"] * 0.5
)
df["colors"] = np.array(color_list)[df["step_id"] % len(color_list)]
df["linewidths"] = 5
```

```python

fig = plt.figure(figsize=(15, 18), constrained_layout=True)
plt.suptitle("Simulation timeline")
gs = GridSpec(7, 1, figure=fig)
plt.subplot(gs[0, 0])
ts = np.linspace(min_time, max_time, 500)
ongoing = []
for t in ts:
    ongoing.append(
        (((df["adjusted_sim_start_time"] < t) & (df["adjusted_sim_done_time"] > t)) * df['sims_per_sec']).sum()
    )
ongoing = np.array(ongoing)
max_serial_sims_per_sec = 1e9 / min_runtime_per_sim_ns
plt.axhline(16, color="k", linestyle="--", label="16x")
plt.axhline(8, color="k", linestyle="--", label="8x")
plt.plot(ts, ongoing / max_serial_sims_per_sec, "k-", label="Parallelism")
plt.legend()
plt.xlim([min_time, max_time])

plt.subplot(gs[1:, 0])
plt.eventplot(
    df["positions"].values[:, None],
    lineoffsets="packet_lineoffsets",
    linelengths="packet_linelengths",
    linewidths="packet_linewidths",
    colors="colors",
    orientation="vertical",
    data=df,
)
plt.eventplot(
    df["positions"].values[:, None],
    lineoffsets="lineoffsets",
    linelengths="linelengths",
    linewidths="linewidths",
    colors="colors",
    orientation="vertical",
    data=df,
)

new_step_reports["adjusted_step_start_time"] = (
    new_step_reports["time"] - new_step_reports["runtime_total"] - offset
)
new_step_reports["adjusted_step_done_time"] = new_step_reports["time"] - offset
for step_id in range(min_step, max_step + 1):
    try:
        rpt = new_step_reports.loc[step_id]
    except KeyError:
        continue
    plt.plot(
        [
            rpt["adjusted_step_start_time"],
            rpt["adjusted_step_done_time"],
        ],
        [-1, -1],
        color_list[step_id % len(color_list)],
        linestyle="--",
        linewidth=5,
    )

for step_id, step_df in df.groupby("step_id"):
    plt.axvline(
        step_df["adjusted_start_time"].min(),
        color="k",
        linestyle="--",
        linewidth=1,
        zorder=100,
    )
    plt.axvline(
        step_df["adjusted_done_time"].max(), color="k", linestyle="--", linewidth=1
    )
for step_id, step_df in df.groupby("step_id"):
    plt.text(
        step_df["adjusted_start_time"].min(),
        -3.7,
        "$\\textbf{Step " + str(step_id) + " submit}$",
        rotation=90,
        va="bottom",
        ha="right",
        color=color_list[step_id % len(color_list)],
        bbox=dict(facecolor="w", edgecolor="w", boxstyle="round"),
    )
    plt.text(
        step_df["adjusted_done_time"].max(),
        df["positions"].max() + 0.3,
        f"Step {step_id} done",
        rotation=90,
        va="bottom",
        ha="right",
        color=color_list[step_id % len(color_list)],
    )
plt.xlabel("Seconds from start")
plt.xlim([min_time, max_time])
plt.ylim([-4, df["positions"].max() + 3.5])
plt.gca().get_yaxis().set_visible(False)

plt.show()
```

```python
done = df.groupby('step_id')['adjusted_done_time'].max()
start = df.groupby('step_id')['adjusted_start_time'].max()
step_prep_time = start.values[3:] - done.values[1:-2]
plt.plot(done.index[3:], step_prep_time)
plt.show()
```

```python
parallelism1 = df["n_total_sims"].sum() * min_runtime_per_sim_ns * 1e-9 / (
    df["sim_done_time"].max() - df["sim_start_time"].min()
)

sim_runtime = working_reports["runtime_simulating"].sum()
total_runtime = 2 * (report_df["done_time"].max() - report_df["start_time"].min())
parallelism2 = sim_runtime / total_runtime
parallelism1, parallelism2, sim_runtime / 3600, total_runtime / 3600

```

```python
print(working_reports['profile'].iloc[0])
```

## Broad table exploration


```python
n_rows_df = pd.DataFrame(
    [
        (table, query(f"select count(*) from {table}").iloc[0][0])
        for table in ch.all_tables
    ],
    columns=["table", "count"],
).set_index("table")
n_rows_df

```

```python
from confirm.adagrid.const import MAX_STEP

n_sims = query("select sum(K) from results").iloc[0][0]
n_retained_sims = query(
    f"select sum(K) from results where inactivation_step={MAX_STEP} and id not in (select id from done where active=false)"
).iloc[0][0]
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
query(
    f"select K, count(*) as n_tiles from tiles where inactivation_step={MAX_STEP} group by K order by K"
)

```

```python
volume_sql = str(2**dim) + "*" + ("*".join([f"radii{d}" for d in range(dim)]))

```

```python
smallest_tile = query(
    f"select * from results where inactivation_step={MAX_STEP} order by {volume_sql} limit 1"
)
smallest_tile

```

```python
largest_tile = query(
    f"select * from results where inactivation_step={MAX_STEP} order by {volume_sql} desc limit 1"
)
largest_tile

```

```python
n_possible_tiles = query("select count(*) from results where isNotNull(lams)").iloc[0][
    0
]
n_possible_tiles

```

```python
lamss_tile = query(
    f"""
select * from results 
    where inactivation_step={MAX_STEP} 
        and isNotNull(lams) 
    order by lams 
    limit 1
"""
)
lamss_tile

```

```python
lamss = lamss_tile["lams"].iloc[0]

```

```python
lams = query(
    f"""
select lams from results 
    where inactivation_step={MAX_STEP} 
        and lams <= {lamss}
"""
)
max_display = lamss * 2
plt.hist(lams["lams"], bins=np.linspace(lamss, max_display, 100))
plt.show()

```

## Ordering


```python
worst5000_df = self.con.query(
    "select * from results where inactivation_step>10000 order by lams limit 5000"
).df()

```

```python
orderer_df = self.con.query(
    f"select orderer from results where inactivation_step>10000 and orderer <= {worst5000_df['orderer'].max()} order by orderer"
).df()

```

```python
_wait = np.searchsorted(orderer_df["orderer"], worst5000_df["orderer"])
_wait

```

```python
cfg = self.get_config().iloc[0].to_dict()

```

```python
import confirm.adagrid.calibrate as adacal
import confirm.models.wd41 as wd41

report = dict()
algo = adacal.AdaCalibrate(
    wd41.WD41(0, 600000, ignore_intersection=True), None, self, cfg, None
)

```

```python
tiles_df = self.next(10, 11, 50000, "orderer")
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

## Looking at the reports


```python
runtime_mins = (working_reports['done_time'].max() - working_reports['start_time'].min()) / 60
runtime_mins
```

```python
plt.plot(working_reports["runtime_per_sim_ns"])
plt.ylim([0, np.percentile(working_reports["runtime_per_sim_ns"], 99.5)])
plt.show()

```

```python
working_reports["runtime_simulating"].sum()

```

```python
new_step_reports
```

```python
new_step_reports['n_deepen'].sum() / 1e6
```

```python
new_step_reports['n_refine'].sum() / 1e6
```

```python
plt.figure(figsize=(10, 10), constrained_layout=True)
plt.subplot(2,2,1)
plt.plot(new_step_reports["lamss"])
plt.xlabel("Step")
plt.ylabel("$\lambda^{**}$")
plt.subplot(2,2,2)
plt.plot(new_step_reports["tie_{k}(lamss)"])
plt.axhline(0.025, color="red", linestyle="--")
plt.xlabel("Step")
plt.ylabel("$\hat{f}(\lambda^{**})$")
plt.ylim([0, 0.03])

plt.subplot(2,2,3)
plt.plot(new_step_reports["grid_cost"])
plt.xlabel("Step")
plt.ylabel(r"$\alpha - \alpha_0$")

plt.subplot(2,2,3)
plt.plot(new_step_reports["grid_cost"], label=r'$\alpha - \alpha_0$')
plt.plot(new_step_reports["bias_tie"], label="bias")
plt.legend()
plt.xlabel("Step")
plt.ylabel(r"$TIE$")

plt.subplot(2,2,4)
plt.plot(new_step_reports["n_refine"], label='$N_{Refine}$)')
plt.plot(new_step_reports["n_deepen"], label="$N_{Deepen}$")
plt.legend()
plt.xlabel("Step")
plt.ylabel("N")
plt.show()
```

```python
import numpy as np

xs = np.linspace(-2, 1, 10)
ys = np.linspace(-2, 1, 10)
counts = np.empty((len(xs), len(ys)))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        NN = self.con.query(
            f"""
            select count(*) 
                from results
                where 
                    inactivation_step={MAX_STEP}
                    and abs(theta0 - {x}) < 0.167
                    and abs(theta2 - {y}) < 0.167
        """
        ).fetchone()[0]
        counts[i, j] = NN

```

```python
counts.sum() / 1e6

```

```python
XX, YY = np.meshgrid(xs, ys, indexing="ij")
# plt.scatter(XX.ravel(), YY.ravel(), c=counts.ravel())
plt.contourf(XX, YY, counts, levels=20)
plt.xlabel("$\\theta_{TNBC, c}$")
plt.ylabel("$\\theta_{HR+, c}$")
cbar = plt.colorbar()
cbar.set_label("$N$")
plt.show()

```

```python
plot_df = self.con.query(
    """
    select theta0, theta1, theta2, theta3, 
            radii0, radii1, radii2, radii3, 
            alpha0, K, lams, twb_mean_lams, twb_min_lams 
        from results
        where 
            abs(theta0 + 1) < 0.05
            and abs(theta2 + 1) < 0.05
"""
).df()

```

```python
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(plot_df["theta1"], plot_df["theta3"], c=plot_df["K"], s=5)
plt.colorbar()
plt.subplot(2, 2, 2)
plt.scatter(plot_df["theta1"], plot_df["theta3"], c=plot_df["alpha0"], s=5)
plt.colorbar()
plt.subplot(2, 2, 3)
plt.scatter(plot_df["theta1"], plot_df["theta3"], c=plot_df["lams"], s=5)
plt.colorbar()
plt.subplot(2, 2, 4)
plt.scatter(plot_df["theta1"], plot_df["theta3"], c=plot_df["twb_min_lams"], s=5)
plt.colorbar()
plt.show()

```
