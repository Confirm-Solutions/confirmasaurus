```python
from imprint.nb_util import setup_nb
setup_nb()
import asyncio
import redis
import numpy as np

import confirm.cloud.clickhouse as ch

from confirm.adagrid.validation import AdaValidate
import confirm.adagrid.adagrid as adagrid
import imprint as ip
from imprint.models.ztest import ZTest1D

import clickhouse_connect
clickhouse_connect.common.set_setting('autogenerate_session_id', False)
```

```python
def random_assignments(seed, n_tiles, n_workers, names):
    # Randomly assign tiles to packets.
    np.random.seed(seed)
    splits = np.array_split(np.arange(n_tiles), n_workers)
    assignment = np.empty(n_tiles, dtype=np.int32)
    for i in range(n_workers):
        assignment[splits[i]] = names[i]
    rng = np.random.default_rng()
    rng.shuffle(assignment)
    return assignment
```

### Glossary

- **coordination**: All workers come together and divide up all tiles.
- **step**: A worker checks the convergence criterion, select tiles, and simulates.
- **packet**: The unit of simulation work that a worker requests from the DB.
- **batch**: The unit of simulation work that is passed to a Model.

```python
# glossary

db = ch.Clickhouse.connect()
g = ip.cartesian_grid(
    theta_min=[-1], theta_max=[1], n=[50000], null_hypos=[ip.hypo("x0 < 0")]
)
kwargs = dict(
    lam=-1.96,
    model_seed=0,
    model_kwargs=None,
    delta=0.01,
    init_K=2**13,
    n_K_double=4,
    tile_batch_size=64,
    max_target=0.002,
    global_target=0.005,
    n_steps = 100,
    step_size=2**10,
    n_iter=1000,
    packet_size = None,
    prod = True,
    overrides = None,
    callback=adagrid.print_report,
    backend=adagrid.LocalBackend(),
    
    model_type=ZTest1D
)
assignment_seed = 1
n_workers = 2

redis_con = db.redis_con
job_id = db.job_id
initial_worker_ids = [redis_con.incr(f"{job_id}:worker_id") + 1 for _ in range(n_workers)]

# TODO: the assignment_seed should change with each coordination.
g.df['worker_assignment'] = random_assignments(assignment_seed, g.n_tiles, n_workers, initial_worker_ids)

packet_size = 2000
import pandas as pd
def assign_packets(df):
    return pd.Series(np.floor(np.arange(df.shape[0]) / packet_size).astype(int), df.index)
g.df['packet_id'] = g.df.groupby('worker_assignment')['worker_assignment'].transform(assign_packets)

ada = adagrid.Adagrid(ZTest1D, g, db, AdaValidate, print, None, kwargs, worker_id = 1)
```

```python
db.get_tiles()
```

```python

```

```python
class NewAdagrid:
    def __init__(self, worker_id):
        self.first_insert = True
        self.worker_id = worker_id

    async def process_step(self, step_id):
        next_work = None
        packet_id = 0
        insert_threads = []
        packet_id = 0
        while True:
            report = dict()
            report['worker_id'] = self.worker_id
            report['step_id'] = step_id
            report['packet_id'] = packet_id

            ########################################
            # Get work
            ########################################
            def get_work():
                return ch._query_df(
                    db.client,
                    f"""
                    select * from tiles
                        where
                            worker_assignment = {self.worker_id}
                            and step_id = {step_id}
                            and packet_id = {packet_id}
                    """,
                )

            start = time.time()
            # On the first loop, we won't have queued any work queries yet.
            if next_work is None:
                next_work = asyncio.to_thread(get_work)
            work = await next_work
            report['n_tiles'] = work.shape[0]

            # Empty packet is an indication that we are done with this step.
            if work.shape[0] == 0:
                report['status'] = 'PACKET_DONE'
                insert_threads.append(asyncio.to_thread(db.insert_report, report))
                report['runtime_done_wait'] = time.time() - start
                await asyncio.gather(*insert_threads)
                break

            # Queue a query for the next packet.
            next_work = asyncio.to_thread(get_work)

            # Check if some other worker has already inserted this packet.
            packet_flag = f"{job_id}:worker_{self.worker_id}_step_{step_id}_packet_{packet_id}_insert"
            flag = redis_con.get(packet_flag)
            if flag is not None:
                logger.debug(
                    "Skipping packet. Flag "
                    f"{packet_flag} is set by worker_id={flag.decode('ascii')}."
                )
                packet_id += 1
                report["runtime_skip_packet"] = time.time() - start
                report["status"] = 'SKIPPED'
                insert_threads.append(asyncio.to_thread(db.insert_report, report))
                continue
            report["runtime_get_work"] = time.time() - start

            ########################################
            # Process tiles
            ########################################
            start = time.time()
            results_df = ada.algo.process_tiles(tiles_df=work, report=report)
            report["runtime_process_tiles"] = time.time() - start

            ########################################
            # Insert results
            ########################################
            start = time.time()
            was_flag_set = redis_con.setnx(packet_flag, self.worker_id)
            if was_flag_set == 0:
                logger.warning(
                    f"(step_id={step_id}, packet_id={packet_id})"
                    " already inserted, discarding results."
                )
                report['status'] = 'DISCARDED'
            else:
                if self.first_insert:
                    # Do the first insert synchronously to make sure the results table is created.
                    db.insert_results(results_df, "total_cost_order, tie_bound_order")
                    self.first_insert = False
                else:
                    insert_threads.append(
                        asyncio.to_thread(
                            db.insert_results,
                            results_df,
                            "total_cost_order, tie_bound_order",
                        )
                    )
                logger.debug(
                    "inserted packet results for "
                    f"(step_id = {step_id}, packet_id={packet_id})"
                    f" with {results_df.shape[0]} results"
                )
                report['status'] = 'WORK'
            insert_threads.append(asyncio.to_thread(db.insert_report, report))
            report["runtime_insert_results"] = time.time() - start
            packet_id += 1
```

```python
import imprint.log
worker_id = initial_worker_ids[0]
imprint.log.worker_id.set(worker_id)

ada_new = NewAdagrid(worker_id)
async with HeartbeatThread(worker_id):
    await ada_new.process_step(0)
    print("finished step")
```

```python
def worst_tile(db, worker_id, order_col):
    return ch._query_df(
        db.client,
        f"""
        select * from results r
            where
                active=true
                and id not in (select id from inactive)
                and worker_assignment = {worker_id}
        order by {order_col} limit 1
    """,
    )
# db.n_processed_tiles(0)
# ada.algo.convergence_criterion
```

```python
def new_step(self, tiles_df, new_step_id, report):
    tiles_df["finisher_id"] = self.cfg["worker_id"]
    tiles_df["active"] = ~(tiles_df["refine"] | tiles_df["deepen"])

    # Record what we decided to do.
    if "split" not in tiles_df.columns:
        tiles_df["split"] = False
    done_cols = [
        "id",
        "step_id",
        "step_iter",
        "active",
        "finisher_id",
        "refine",
        "deepen",
        "split",
    ]
    # TODO: call finish in separate thread to avoid blocking
    self.db.finish(tiles_df[done_cols])

    n_refine = tiles_df["refine"].sum()
    n_deepen = tiles_df["deepen"].sum()
    report.update(
        dict(
            n_refine=n_refine,
            n_deepen=n_deepen,
            n_complete=tiles_df["active"].sum(),
        )
    )

    nothing_to_do = n_refine == 0 and n_deepen == 0
    if nothing_to_do:
        return "empty"

    # Actually deepen and refine!
    g = refine_and_deepen(
        tiles_df, self.null_hypos, self.cfg["max_K"], self.cfg["worker_id"]
    )
    g.df["step_id"] = new_step_id
    g.df["creator_id"] = self.cfg["worker_id"]
    g.df["creation_time"] = imprint.timer.simple_timer()

    # there might be new inactive tiles that resulted from splitting with
    # the null hypotheses. we need to mark these tiles as finished.
    # TODO: inactive insert_tiles and finish can be done in separate thread
    # to avoid blocking
    inactive_df = g.df[~g.df["active"]].copy()
    inactive_df["step_iter"] = np.int32(-1)
    self.db.insert_tiles(inactive_df)
    inactive_df["refine"] = False
    inactive_df["deepen"] = False
    inactive_df["split"] = True
    inactive_df["finisher_id"] = self.cfg["worker_id"]
    self.db.finish(inactive_df[done_cols])

    # Assign tiles to packets and then insert them into the database for
    # processing.
    g_active = g.prune_inactive()
    g_active.df["step_iter"], n_packets = step_iter_assignments(
        g_active.df, self.cfg["packet_size"]
    )
    # TODO: i think insert_tiles can be done in separate thread to avoid
    # blocking
    self.db.insert_tiles(g_active.df)
    self.db.set_step_info(
        step_id=new_step_id, step_iter=0, n_iter=n_packets, n_tiles=g_active.n_tiles
    )

    logger.debug(
        f"new step {(new_step_id, 0, n_packets, g.n_tiles)}\n"
        f"n_tiles={g_active.n_tiles} packet_size={self.cfg['packet_size']}\n"
        f"n_inactive_tiles={inactive_df.shape[0]}"
    )
    report.update(
        dict(
            n_new_tiles=g.n_tiles,
            new_K_distribution=g.df["K"].value_counts().to_dict(),
        )
    )
    return new_step_id
```

```python
# TODO: if n_processed_tiles == step_n_tiles:
report = dict()
step_id = 0
ada.algo.c["worker_id"] = worker_id
start = time.time()
converged, convergence_data = ada.algo.convergence_criterion(report)
report["runtime_convergence_criterion"] = time.time() - start
if converged:
    report["status"] = "CONVERGED"
    logger.debug("Convergence!!")
    # TODO: return
elif step_id >= ada.algo.c["n_steps"] - 1:
    report["status"] = "MAX_STEPS"
    print("max steps")
    # TODO: end the job.
else:
    # If we haven't converged, we create a new step.
    start = time.time()
    new_step_id = step_id + 1
    tiles_df = ada.algo.select_tiles(report, convergence_data)
    report["runtime_select_tiles"] = time.time() - start

    if tiles_df is None:
        # New step is empty so we have terminated but
        # failed to converge.
        logger.debug(
            "New packet is empty. Waiting for the next "
            "coordination despite failure to converge."
        )
        report['status'] = 'EMPTY_STEP'
        # TODO: wait for coordination
    else:
        report["status"] = "NEW_STEP"
        report["n_tiles"] = tiles_df.shape[0]

```

```python
report = dict()
max_tie_est = worst_tile(db, worker_id, "tie_est desc")["tie_est"].iloc[0]
next_tile = worst_tile(db, worker_id, "total_cost_order, tie_bound_order").iloc[0]
report['converged'] = ada.algo._are_tiles_done(next_tile, max_tie_est)
report.update(
    dict(
        max_tie_est=max_tie_est,
        next_tile_tie_est=next_tile["tie_est"],
        next_tile_tie_bound=next_tile["tie_bound"],
        next_tile_sim_cost=next_tile["sim_cost"],
        next_tile_grid_cost=next_tile["grid_cost"],
        next_tile_total_cost=next_tile["total_cost"],
        next_tile_K=next_tile["K"],
        next_tile_at_max_K=next_tile["K"] == ada.algo.max_K,
    )
)
report['converged'], max_tie_est
```

```python
report
```

```python
db.get_reports()
```

```python

```
