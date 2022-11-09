---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.10.6 ('base')
    language: python
    name: python3
---

```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()
import jax
import numpy as np
import jax.numpy as jnp
from confirm.lewislib import lewis

# Configuration used during simulation
name = "4d_full"
params = {
    "n_arms": 4,
    "n_stage_1": 50,
    "n_stage_2": 100,
    "n_stage_1_interims": 2,
    "n_stage_1_add_per_interim": 100,
    "n_stage_2_add_per_interim": 100,
    "stage_1_futility_threshold": 0.15,
    "stage_1_efficacy_threshold": 0.7,
    "stage_2_futility_threshold": 0.2,
    "stage_2_efficacy_threshold": 0.95,
    "inter_stage_futility_threshold": 0.6,
    "posterior_difference_threshold": 0,
    "rejection_threshold": 0.05,
    "key": jax.random.PRNGKey(0),
    "n_table_pts": 20,
    "n_pr_sims": 100,
    "n_sig2_sims": 20,
    "batch_size": int(2**12),
    "cache_tables": f"./{name}/lei_cache.pkl",
}
lei_obj = lewis.Lewis45(**params)
n_arm_samples = int(lei_obj.unifs_shape()[0])
```

```python
import confirm.mini_imprint.lewis_drivers as lts

N = 256
K = 2**18
unifs = jax.random.uniform(jax.random.PRNGKey(0), (K, n_arm_samples, 4))
unifs_order = jnp.arange(n_arm_samples)
theta = np.random.rand(N, 4)
null_truth = np.ones((N, 3), dtype=bool)
sim_sizes = np.full(N, 2**13)
lam = 0.05
```

## Is copying the uniforms array expensive?

```python
%%time
unifs.sum(axis=1).block_until_ready()
```

```python
%%time
outs = []
for i in range(0, unifs.shape[0], 10000):
    begin_idx = i
    end_idx = min(i + 10000, unifs.shape[0])
    outs.append(unifs[begin_idx:end_idx].sum(axis=1))
outs = jnp.concatenate(outs).block_until_ready()
```

```python
%%time
unifs[:100000].sum(axis=1).block_until_ready()
```

```python
%%time
outs = []
for i in range(0, 100000, 10000):
    begin_idx = i
    end_idx = min(i + 10000, unifs.shape[0])
    outs.append(unifs[begin_idx:end_idx].sum(axis=1))
outs = jnp.concatenate(outs).block_until_ready()
```

```python
%%time
unifs[:30000].sum(axis=1).block_until_ready()
```

```python
%%time
outs = []
for i in range(0, 30000, 10000):
    begin_idx = i
    end_idx = min(i + 10000, unifs.shape[0])
    outs.append(unifs[begin_idx:end_idx].sum(axis=1))
outs = jnp.concatenate(outs).block_until_ready()
```

```python
import gc

gc.collect()
```

```python
lts.memory_status("hi")
```

## What batch sizes??

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=8192,
    grid_batch_size=64,
)
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=8192,
    grid_batch_size=512,
)
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=1024,
    grid_batch_size=64,
)
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=2048,
    grid_batch_size=128,
)
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=8192,
    grid_batch_size=128,
)
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=512,
    grid_batch_size=128,
)
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=64,
    grid_batch_size=64,
)
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes[:64],
    lam,
    theta[:64],
    null_truth[:64],
    unifs,
    unifs_order,
    sim_batch_size=1024,
    grid_batch_size=64,
)
```

```python
import time
import gc
import jax.numpy as jnp
from confirm.lewislib import batch
from confirm.mini_imprint.lewis_drivers import get_sim_size_groups


def simulator(p, unifs, unifs_order):
    return jnp.sum(unifs[:, :] < p[None, :]) / unifs.size, 1, 0


simulatev = jax.vmap(simulator, in_axes=(None, 0, None))


def stat(lei_obj, theta, null_truth, unifs, unifs_order):
    p = jax.scipy.special.expit(theta)
    test_stats, best_arms, _ = simulatev(p, unifs, unifs_order)
    false_test_stats = jnp.where(null_truth[best_arms - 1], test_stats, 100.0)
    return false_test_stats


statv = jax.jit(jax.vmap(stat, in_axes=(None, 0, 0, None, None)), static_argnums=(0,))


@jax.jit
def sumstats(stats, lam):
    return jnp.sum(stats < lam, axis=-1)


def rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=1024,
    grid_batch_size=64,
):
    outs = []
    for (_, idx, stats) in _stats_backend(
        lei_obj,
        sim_sizes,
        theta,
        null_truth,
        unifs,
        unifs_order,
        sim_batch_size,
        grid_batch_size,
    ):
        outs.append(sumstats(stats, lam))
    return jnp.concatenate(outs)


def _stats_backend(
    lei_obj,
    sim_sizes,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=1024,
    grid_batch_size=64,
):
    batched_statv = batch.batch(
        batch.batch(
            statv, sim_batch_size, in_axes=(None, None, None, 0, None), out_axes=(1,)
        ),
        grid_batch_size,
        in_axes=(None, 0, 0, None, None),
    )

    for size, idx in get_sim_size_groups(sim_sizes):
        print(
            f"simulating with K={size} and n_tiles={idx.sum()}"
            f" and batch_size=({grid_batch_size}, {sim_batch_size})"
        )
        start = time.time()
        unifs_chunk = unifs[:size]
        stats = batched_statv(
            lei_obj, theta[idx], null_truth[idx], unifs_chunk, unifs_order
        )
        print("simulation runtime", time.time() - start)

        yield (size, idx, stats)
```

```python
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def stat_sum(lei_obj, lam, theta, null_truth, unifs, unifs_order):
    stats = jax.vmap(stat, in_axes=(None, 0, 0, None, None))(
        lei_obj, theta, null_truth, unifs, unifs_order
    )
    return jnp.sum(stats < lam, axis=-1)
```

```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (K, 350, 4))
```

```python
%%time
unifs_chunk = unifs[:1024]
res = stat_sum(
    lei_obj, 0.6, theta, null_truth, unifs_chunk, unifs_order
).block_until_ready()
```

```python
batched_stat_sum = batch.batch(
    batch.batch(
        stat_sum, 8192, in_axes=(None, None, None, None, 0, None), out_axes=(1,)
    ),
    256,
    in_axes=(None, None, 0, 0, None, None),
)
```

```python
%%time
unifs_chunk = unifs[:8192]
stats = batched_stat_sum(
    lei_obj, 0.05, theta, null_truth, unifs[:8192], unifs_order
).block_until_ready()
```

```python
%%time
unifs_chunk = unifs[:8192]
stats = statv(lei_obj, theta, null_truth, unifs_chunk, unifs_order)
rej = jnp.sum(stats < lam, axis=-1).block_until_ready()
```

```python
batched_statv = batch.batch(
    batch.batch(statv, 8192, in_axes=(None, None, None, 0, None), out_axes=(1,)),
    256,
    in_axes=(None, 0, 0, None, None),
)
```

```python
%%time
unifs_chunk = unifs[:8192]
stats = batched_statv(lei_obj, theta, null_truth, unifs_chunk, unifs_order)
rej = jnp.sum(stats < lam, axis=-1).block_until_ready()
```

```python
batched_statv2 = batch.batch(
    batch.batch(statv, 2048, in_axes=(None, None, None, 0, None), out_axes=(1,)),
    128,
    in_axes=(None, 0, 0, None, None),
)
```

```python
%%time
unifs_chunk = unifs[:8192]
stats = batched_statv2(lei_obj, theta, null_truth, unifs_chunk, unifs_order)
rej = jnp.sum(stats < lam, axis=-1).block_until_ready()
```

```python
%%time
unifs_chunk = unifs[:8192]
stats = statv(lei_obj, theta, null_truth, unifs_chunk, unifs_order)
rej = np.sum(stats < lam, axis=-1).block_until_ready()
```

```python
%%time
unifs_chunk = unifs[:1024]
rejs = []
j_step = 128
i_step = 1024
for i in range(0, unifs_chunk.shape[0], i_step):
    i_begin = i
    i_end = i + i_step
    for j in range(0, theta.shape[0], j_step):
        j_begin = j
        j_end = min(theta.shape[0], j + j_step)
        subunifs_chunk = unifs_chunk[i_begin:i_end]
        # rejs.append(stat_sum(lei_obj, lam, theta[j_begin:j_end], null_truth[j_begin:j_end], subunifs_chunk, unifs_order))
        stats = statv(
            lei_obj,
            theta[j_begin:j_end],
            null_truth[j_begin:j_end],
            subunifs_chunk,
            unifs_order,
        )
        rejs.append(jnp.sum(stats < lam, axis=-1))
rej = jnp.concatenate(rejs).block_until_ready()
```

```python
%%time
rejs = lts.rej_runner(
    lei_obj,
    sim_sizes,
    lam,
    theta,
    null_truth,
    unifs,
    unifs_order,
    sim_batch_size=2048,
    grid_batch_size=128,
)
```

```python

```

```python
unifs = jax.random.uniform(jax.random.PRNGKey(10), (K, n_arm_samples, 4))
```

```python
%load_ext line_profiler
```

```python
%lprun -T output.log -f rej_runner -f _stats_backend rej_runner(lei_obj, sim_sizes, lam, theta, null_truth, unifs, unifs_order, sim_batch_size=1024, grid_batch_size=64)
open("output.log").read()
```

```python

```
