---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.10.6 ('confirm')
    language: python
    name: python3
---

# Benchmarking to inform the Model API design.

I had several question that are important to answer to not hamstring our
performance when we start to commit to a Model API:

1. Does slicing and copying a large array of uniforms (or other pregenerated random variates) cause performance issues?
    - No! It does not. This is irrelevant even though I had previously
      suspected this was a significant cause of performance problems. Both the
      microbenchmark and later full benchmark demonstrate this. This
      is true across CPU and GPU.
2. What batch size should we use?
    - Larger when the simulation function is slower. (duh)
    - 1024 sims x 64 pts is reasonable for something like Lei.
    - On CPU: Large-ish but it's actually faster to use some batches rather than
      running the whole thing at once. This is unsurprising and due to
      cache-friendliness. Ideal was 32768 sims and 128 grid points in a single batch.
    - On GPU:
3. Are the concatenations in our current GPU code problematic.
    - On CPU: Yes, concatenation is bad for performance, especially when we're
      double batching over both grid points and simulations since we incur a
      concatenation for each outer batch.
    - On GPU: CHECK THIS! This will be especially bad because we force blocking
      and copy data from GPU to CPU.
4. Does it help to include the summation of rejections inside the jitted function call? Or can we factor that out into the calling code?
    - On CPU, we get better performance when we include the summation inside the jit call.
    - On GPU: CHECK THIS!

```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()

import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from confirm.lewislib import batch
```

## Is copying the uniforms array expensive?



```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (256000, 350, 4))
unifs10 = jax.random.uniform(jax.random.PRNGKey(0), (10000, 350, 4))
unifs2d = jax.random.uniform(jax.random.PRNGKey(0), (256000, 350, 1))
```

```python
for k in range(4):
    start = time.time()
    out1 = unifs.sum(axis=1).block_until_ready()
    if k >= 1:
        print("sum", time.time() - start)

    start = time.time()
    copy = unifs.copy().block_until_ready()
    if k >= 1:
        print("copy", time.time() - start)

    start = time.time()
    out2 = unifs[:-1].sum(axis=1).block_until_ready()
    if k >= 1:
        print("slicesum", time.time() - start)

    start = time.time()
    outs = []
    for i in range(0, unifs.shape[0], 10000):
        begin_idx = i
        end_idx = min(i + 10000, unifs.shape[0])
        outs.append(unifs[begin_idx:end_idx].sum(axis=1))
    out2 = jnp.concatenate(outs).block_until_ready()
    if k >= 1:
        print("batched sum", time.time() - start)
    np.testing.assert_allclose(out1, out2)

    start = time.time()
    unifs10.sum(axis=1).block_until_ready()
    if k >= 1:
        print("sum10k no slice", time.time() - start)

    start = time.time()
    unifs[:10000].sum(axis=1).block_until_ready()
    if k >= 1:
        print("sum10k with slice", time.time() - start)

    start = time.time()
    unifs2d.sum(axis=1).block_until_ready()
    if k >= 1:
        print("sum2d no slice", time.time() - start)

    start = time.time()
    unifs[:, :, :1].sum(axis=1).block_until_ready()
    if k >= 1:
        print("sum2d with slice", time.time() - start)
```

## What batch sizes??

```python
def simulator(p, unifs):
    return jnp.sum(unifs[:, :] < p[None, :]) / unifs.size


def stat(theta, null_truth, unifs):
    p = jax.scipy.special.expit(theta)
    simulatev = jax.vmap(simulator, in_axes=(None, 0))
    test_stats = simulatev(p, unifs)
    false_test_stats = jnp.where(null_truth[0], test_stats, 100.0)
    return false_test_stats


statv = jax.jit(jax.vmap(stat, in_axes=(0, 0, None)))


@partial(jax.jit, static_argnums=(0,))
def stat_sum(lam, theta, null_truth, unifs):
    stats = jax.vmap(stat, in_axes=(0, 0, None))(theta, null_truth, unifs)
    return jnp.sum(stats < lam, axis=-1)
```

```python
N = 1024
theta = np.random.rand(N, 4)
null_truth = np.ones((N, 3), dtype=bool)
sim_sizes = np.full(N, 2**13)
lam = 0.05
```

In the cell below, I'm comparing several things:
1. How does the batch size affect the output? 
    - caution: I think this is quite different on GPU versus CPU. JAX on GPU
      pipelines GPU calls as long as we don't block and wait for the results.
      So, smaller batches are acceptable on the GPU.    

```python
def simple(unifs_chunk, _1, _2):
    return stat_sum(
        lam,
        theta,
        null_truth,
        unifs_chunk,
    ).block_until_ready()


def batched(unifs_chunk, sim_batch_size, grid_batch_size):
    batched_stat_sum = batch.batch(
        batch.batch(stat_sum, sim_batch_size, in_axes=(None, None, None, 0)),
        grid_batch_size,
        in_axes=(None, 0, 0, None),
    )
    return batched_stat_sum(lam, theta, null_truth, unifs_chunk)


def late_concat_batch(unifs_chunk, sim_batch_size, grid_batch_size):
    rejs = []
    for i in range(0, unifs_chunk.shape[0], sim_batch_size):
        i_begin = i
        i_end = min(unifs_chunk.shape[0], i + sim_batch_size)
        for j in range(0, theta.shape[0], grid_batch_size):
            j_begin = j
            j_end = min(theta.shape[0], j + grid_batch_size)
            subunifs_chunk = unifs_chunk[i_begin:i_end]
            rejs.append(
                stat_sum(
                    lam,
                    theta[j_begin:j_end],
                    null_truth[j_begin:j_end],
                    subunifs_chunk,
                )
            )
    return jnp.concatenate(rejs).block_until_ready()


def late_concat_stat_then_sum(unifs_chunk, sim_batch_size, grid_batch_size):
    rejs = []
    for i in range(0, unifs_chunk.shape[0], sim_batch_size):
        i_begin = i
        i_end = min(unifs_chunk.shape[0], i + sim_batch_size)
        for j in range(0, theta.shape[0], grid_batch_size):
            j_begin = j
            j_end = min(theta.shape[0], j + grid_batch_size)
            subunifs_chunk = unifs_chunk[i_begin:i_end]
            stats = statv(
                theta[j_begin:j_end],
                null_truth[j_begin:j_end],
                subunifs_chunk,
            )
            rejs.append(jnp.sum(stats < lam, axis=-1))
    return jnp.concatenate(rejs).block_until_ready()
```

```python
fncs = dict(
    simple=simple,
    batched=batched,
    late_concat_batch=late_concat_batch,
    late_concat_stat_then_sum=late_concat_stat_then_sum,
)
```

```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (32768, 1, 4))

run_keys = list(fncs.keys())
for sim_batch_size in [1024, 2048, 4096, 8192, 16384, 32768]:
    print(" ")
    for grid_batch_size in [32, 64, 128, 256, 512, 1024]:
        print(" ")
        for k in range(2):
            for run_key in run_keys:
                start = time.time()
                result = fncs[run_key](unifs, sim_batch_size, grid_batch_size)
                if k >= 1:
                    print(
                        f"{run_key} ({sim_batch_size}, {grid_batch_size}) = {time.time() - start}"
                    )
```

```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (16384, 5, 4))

run_keys = list(fncs.keys())
for sim_batch_size in [512, 1024, 2048, 4096, 8192, 16384]:
    print(" ")
    for grid_batch_size in [32, 64, 128, 256, 512, 1024]:
        print(" ")
        for k in range(2):
            for run_key in run_keys:
                start = time.time()
                result = fncs[run_key](unifs, sim_batch_size, grid_batch_size)
                if k >= 1:
                    print(
                        f"{run_key} ({sim_batch_size}, {grid_batch_size}) = {time.time() - start}"
                    )
```

```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (16384, 31, 4))

run_keys = list(fncs.keys())
for sim_batch_size in [512, 1024, 2048, 4096, 8192, 16384]:
    print(" ")
    for grid_batch_size in [32, 64, 128, 256, 512, 1024]:
        print(" ")
        for k in range(2):
            for run_key in run_keys:
                start = time.time()
                result = fncs[run_key](unifs, sim_batch_size, grid_batch_size)
                if k >= 1:
                    print(
                        f"{run_key} ({sim_batch_size}, {grid_batch_size}) = {time.time() - start}"
                    )
```

```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (16384, 150, 4))

fncs["simple"](unifs, 1, 1)

start = time.time()
fncs["simple"](unifs, 1, 1)
print(time.time() - start)

run_keys = ["batched", "late_concat_batch", "late_concat_stat_then_sum"]
for sim_batch_size in [2048, 4096, 8192, 16384]:
    print(" ")
    for grid_batch_size in [128, 256, 512, 1024]:
        print(" ")
        for k in range(2):
            for run_key in run_keys:
                start = time.time()
                result = fncs[run_key](unifs, sim_batch_size, grid_batch_size)
                if k >= 1:
                    print(
                        f"{run_key} ({sim_batch_size}, {grid_batch_size}) = {time.time() - start}"
                    )
```

```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (16384, 150, 4))

run_keys = ["batched", "late_concat_batch", "late_concat_stat_then_sum"]
for sim_batch_size in [256, 512, 1024]:
    print(" ")
    for grid_batch_size in [32, 64, 128, 256]:
        print(" ")
        for k in range(2):
            for run_key in run_keys:
                start = time.time()
                result = fncs[run_key](unifs, sim_batch_size, grid_batch_size)
                if k >= 1:
                    print(
                        f"{run_key} ({sim_batch_size}, {grid_batch_size}) = {time.time() - start}"
                    )
```

```python

```
