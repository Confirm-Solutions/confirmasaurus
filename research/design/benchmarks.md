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

# Benchmarking to inform the Model API design.

I had several question that are important to answer to not hamstring our
performance when we start to commit to a Model API (https://github.com/Confirm-Solutions/confirmasaurus/pull/92):

1. Does slicing and copying a large array of uniforms (or other pregenerated random variates) cause performance issues?
   - See the end of the doc.
   - No! It does not. This is irrelevant even though I had previously
     suspected this was a significant cause of performance problems. Both the
     microbenchmark and later full benchmark demonstrate this. This is true
     across CPU and GPU.
2. What batch size should we use?
     - when the problem is memory bandwidth constrained (simple binomial with n_arm_samples getting large), there's a narrow band of optimal performance.
       - see the figures below for n_arm_samples = 128 and n_arm_samples = 350. These are severely memory constrained cases because all we do with each uniform samples is do a less than comparison against p --> one operation per number.
     - when the model is compute constrained or very small/fast, we should use the largest batches that we can. This will be mostly constrained by running out of memory in something like the `unifs < p` call.
     - for Lei, the optimal was 1024 x as many grid points as possible
3. Are the concatenations in our current GPU code problematic?
   - No. We should use concatenation because it's simpler and not slower! This
     is great news and makes it easy to have a clean `batch(...)` code.
   - Compare `batched_flip` to `batched_flip_noconcat` below.
4. Does it help to include the summation of rejections inside the jitted
   function call? Or can we factor that out into the calling code? It's nice to factor this out because then the Model API can just return test statistics and not be responsible for computing the number of rejections or doing the tuning.
   - it is sometimes faster to include the summation inside the jitted
     function call and sometimes slower. I am a bit confused about the behavior here but I'm guessing it has to do with the details of JAX compilation and how it is or isn't reordering operation.
   - to generalize:
     - for very very fast simulators, it's faster to include the summation
       inside the JIT call. The difference is 10-40% performance.
     - for slower simulations, it doesn't matter whether we include the summation inside the JIT and sometimes including the summation inside the JIT slows down overall 
   - compare `sum_stat` (sum inside JIT) to `sum_then_stat` (sum separate from main Model JIT) below.
5. What order should we batch in? Simulations outside and points inside or vice versa? 
	- for memory constrained cases like this binomial setting where there are lots of pregenerated uniforms that are being summed, it's much faster for the outer batch to be over simulations and the inner batch to be over simulation points.
	- In compute constrained cases, the ordering doesn't seem matter so we should tend to have the inner batch be over simulation points

None of this matters very much when applied to Lei. Except the bug I unearthened where we were copying uniforms to and from the GPU accidentally!


```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()

import time
import timeit
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from confirm.lewislib import batch
import confirm.mini_imprint.lewis_drivers as ld
from confirm.lewislib import lewis
```

## Optimizing the batching code


```python
def simulator(p, unifs):
    return jnp.sum(unifs[:, :] < p[None, :]) / unifs.size


@jax.jit
def stat_sum(lam, theta, null_truth, unifs):
    p = jax.scipy.special.expit(theta)
    test_stats = jnp.sum(unifs[None, :, :, :] < p[:, None, None, :], axis=(2, 3)) / (
        unifs.shape[1] * unifs.shape[2]
    )
    rej = test_stats < lam
    return jnp.sum(rej * null_truth[:, None, 0], axis=-1)


def stat(theta, null_truth, unifs):
    p = jax.scipy.special.expit(theta)
    simulatev = jax.vmap(simulator, in_axes=(None, 0))
    test_stats = simulatev(p, unifs)
    false_test_stats = jnp.where(null_truth[0], test_stats, 100.0)
    return false_test_stats


statv = jax.jit(jax.vmap(stat, in_axes=(0, 0, None)))


@jax.jit
def sumlam(stats, lam):
    return jnp.sum(stats < lam, axis=-1)


def stat_then_sum(lam, theta, null_truth, unifs):
    stats = statv(theta, null_truth, unifs)
    out = sumlam(stats, lam)
    return out
```

```python
def simple(unifs_chunk, _1, _2):
    return stat_sum(
        lam,
        theta,
        null_truth,
        unifs_chunk,
    ).block_until_ready()


def simple_then_sum(unifs_chunk, _1, _2):
    return stat_then_sum(
        lam,
        theta,
        null_truth,
        unifs_chunk,
    ).block_until_ready()


def batched(unifs_chunk, sim_batch_size, grid_batch_size):
    sim_batcher = batch.batch(
        stat_sum, sim_batch_size, in_axes=(None, None, None, 0), out_axes=(1,)
    )
    batched_stat_sum = batch.batch(
        lambda *args: sim_batcher(*args).sum(axis=-1),
        grid_batch_size,
        in_axes=(None, 0, 0, None),
    )
    return batched_stat_sum(lam, theta, null_truth, unifs_chunk).block_until_ready()


def batched_flip(statter):
    def f(unifs_chunk, sim_batch_size, grid_batch_size):
        batched_stat_sum = batch.batch(
            batch.batch(
                statter, grid_batch_size, in_axes=(None, 0, 0, None), out_axes=(0,)
            ),
            sim_batch_size,
            in_axes=(None, None, None, 0),
            out_axes=(1,),
        )
        res = batched_stat_sum(lam, theta, null_truth, unifs_chunk)
        return res.sum(axis=1).block_until_ready()

    f.__name__ = batched_flip.__name__ + "_" + statter.__name__
    return f


def batched_flip_noconcat(statter):
    def f(unifs_chunk, sim_batch_size, grid_batch_size):
        batched_stat_sum = batch.batch_all(
            batch.batch_all(statter, grid_batch_size, in_axes=(None, 0, 0, None)),
            sim_batch_size,
            in_axes=(None, None, None, 0),
        )
        res, _ = batched_stat_sum(lam, theta, null_truth, unifs_chunk)
        n_j = len(res[0][0])

        def entry(i, j):
            e = res[i][0][j]
            if j == n_j - 1:
                e[: -res[i][1]]
            return e

        return (
            jnp.block([[entry(i, j) for j in range(n_j)] for i in range(len(res))])
            .sum(axis=0)
            .block_until_ready()
        )

    f.__name__ = batched_flip_noconcat.__name__ + "_" + statter.__name__
    return f
```

```python
lam = 0.5
```

```python
fncs = [
    simple,
    simple_then_sum,
    batched,
    batched_flip(stat_sum),
    batched_flip(stat_then_sum),
    batched_flip_noconcat(stat_sum),
    batched_flip_noconcat(stat_then_sum),
]
import timeit

N = 4096
K = 2**17
print(K)
unifs = jax.random.uniform(jax.random.PRNGKey(0), (K, 1, 4))
theta = np.random.rand(N, 4)
null_truth = np.ones((N, 3), dtype=bool)
sim_sizes = np.full(N, 2**13)

sim_batch_size = K // 2
res = simple(unifs, 0, 0)
for f in fncs:
    res_compare = f(unifs, sim_batch_size, 1024)
    np.testing.assert_allclose(res, res_compare)
    print(
        f.__name__,
        min(timeit.repeat(lambda: f(unifs, sim_batch_size, 1024), repeat=15, number=1)),
    )
```

```python
fncs = [
    simple,
    simple_then_sum,
    batched,
    batched_flip(stat_sum),
    batched_flip(stat_then_sum),
    batched_flip_noconcat(stat_sum),
    batched_flip_noconcat(stat_then_sum),
]
import timeit

N = 2048
K = 2**17
print(K)
unifs = jax.random.uniform(jax.random.PRNGKey(0), (K, 5, 4))
theta = np.random.rand(N, 4)
null_truth = np.ones((N, 3), dtype=bool)
sim_sizes = np.full(N, 2**13)

sim_batch_size = K // 4
res = simple(unifs, 0, 0)
for f in fncs:
    res_compare = f(unifs, sim_batch_size, 1024)
    np.testing.assert_allclose(res, res_compare)
    print(
        f.__name__,
        min(timeit.repeat(lambda: f(unifs, sim_batch_size, 1024), repeat=15, number=1)),
    )
```

```python
N = 1024
theta = np.random.rand(N, 4)
null_truth = np.ones((N, 3), dtype=bool)
sim_sizes = np.full(N, 2**13)

grid_batch_size = 256
sim_batch_size = 1024
unifs = jax.random.uniform(jax.random.PRNGKey(0), (16384, 150, 4))

fncs = [
    simple,
    simple_then_sum,
    batched,
    batched_flip(stat_sum),
    batched_flip(stat_then_sum),
    batched_flip_noconcat(stat_sum),
    batched_flip_noconcat(stat_then_sum),
]
import timeit

res = simple(unifs, 0, 0)
for f in fncs:
    res_compare = f(unifs, 1024, 256)
    np.testing.assert_allclose(res, res_compare)
    print(
        f.__name__, min(timeit.repeat(lambda: f(unifs, 1024, 256), repeat=40, number=1))
    )
```

## What batch size??


```python
bench_data = []


def batching_benchmark(
    n_arm_samples, K=32768, S=[4096, 8192, 16384, 32768], G=[128, 256, 512, 1024]
):
    unifs = jax.random.uniform(jax.random.PRNGKey(0), (32768, n_arm_samples, 4))
    fncs = [
        simple,
        batched_flip_noconcat(stat_sum),
        batched_flip_noconcat(stat_then_sum),
    ]

    for sim_batch_size in S:
        for grid_batch_size in G:
            for f in fncs:
                _ = f(unifs, sim_batch_size, grid_batch_size)
                runtime = min(
                    timeit.repeat(
                        lambda: f(unifs, sim_batch_size, grid_batch_size),
                        repeat=10,
                        number=1,
                    )
                )
                d = dict(
                    n_arm_samples=n_arm_samples,
                    sim_batch_size=sim_batch_size,
                    grid_batch_size=grid_batch_size,
                    fnc=f.__name__,
                    time=runtime,
                )
                bench_data.append(d)
```

```python
batching_benchmark(1, G=[512, 1024])
```

```python
batching_benchmark(4, G=[512, 1024])
```

```python
batching_benchmark(8, G=[512, 1024])
```

```python
batching_benchmark(16, G=[512, 1024])
```

```python
batching_benchmark(32, S=[1024, 2048, 4096, 8192], G=[512, 1024])
```

```python
batching_benchmark(64, K=16384, S=[1024, 2048, 4096], G=[512, 1024])
```

```python
batching_benchmark(
    128, K=4096, S=[512, 768, 1024, 1280, 1536, 1792, 2048], G=[64, 128, 256, 512, 1024]
)
```

```python
batching_benchmark(
    350, K=8192, S=[128, 256, 384, 512, 542, 574, 606, 640, 768, 1024], G=[512, 1024]
)
```

```python
df = pd.DataFrame(bench_data)
```

```python
idxmin = df.loc[df["fnc"] != "simple"].groupby("n_arm_samples")["time"].idxmin()
results = df.loc[idxmin].merge(
    df.loc[df["fnc"] == "simple"],
    on=("n_arm_samples", "sim_batch_size", "grid_batch_size"),
)
results
```

```python
res128 = df.loc[
    (df["n_arm_samples"] == 128)
    & (df["fnc"] == "batched_flip_noconcat_stat_sum")
    & (df["grid_batch_size"] == 1024)
].sort_values("sim_batch_size")
res128.plot(x="sim_batch_size", y="time", style=["k-o"])
res128
```

```python
res350 = df.loc[
    (df["n_arm_samples"] == 350)
    & (df["fnc"] == "batched_flip_noconcat_stat_sum")
    & (df["grid_batch_size"] == 1024)
].sort_values("sim_batch_size")
res350.plot(x="sim_batch_size", y="time", style=["k-o"])
res350
```

## Applying this to Lei


```python
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
    "cache_tables": f"./lei_cache.pkl",
}
lei_obj = lewis.Lewis45(**params)
```

```python
N = 1024
theta = np.random.rand(N, 4) - 0.5
null_truth = np.ones((N, 3), dtype=bool)
# null_truth = np.random.rand(N, 3) < 0.5
sim_sizes = np.full(N, 2**13)
lam = 0.5

grid_batch_size = 256
sim_batch_size = 1024
unifs = jax.random.uniform(jax.random.PRNGKey(0), (16384, 350, 4))
unifs_order = jnp.arange(350)
```

```python
# made unifs_order a global variable so that I don't need to change the code
# from above. this is lazy but fine for throwaway code.
def simulator(p, unifs):
    return lei_obj.simulate(p, unifs, unifs_order)[:2]


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, None))
def stat_sum(lam, theta, null_truth, unifs):
    p = jax.scipy.special.expit(theta)
    test_stats, best_arms = jax.vmap(simulator, in_axes=(None, 0))(p, unifs)
    rej = test_stats < lam
    return jnp.sum(rej * null_truth[best_arms - 1])


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None))
def stat(theta, null_truth, unifs):
    p = jax.scipy.special.expit(theta)
    test_stats, best_arms = jax.vmap(simulator, in_axes=(None, 0))(p, unifs)
    false_test_stats = jnp.where(null_truth[best_arms - 1], test_stats, jnp.inf)
    return false_test_stats


@jax.jit
def sumlam(stats, lam):
    return jnp.sum(stats < lam, axis=-1)


def stat_then_sum(lam, theta, null_truth, unifs):
    stats = stat(theta, null_truth, unifs)
    out = sumlam(stats, lam)
    return out
```

```python
def simple(unifs_chunk, _1, _2):
    return stat_sum(
        lam,
        theta,
        null_truth,
        unifs_chunk,
    ).block_until_ready()
```

```python
%%timeit
simple(unifs[:2048], None, None)
```

```python
bench_data = []


def batching_benchmark(
    n_arm_samples, K=32768, S=[128, 256, 512, 1024], G=[256, 512, 1024]
):
    unifs = jax.random.uniform(jax.random.PRNGKey(0), (32768, n_arm_samples, 4))
    fncs = [
        batched_flip(stat_then_sum),
        batched_flip_noconcat(stat_sum),
    ]

    for sim_batch_size in S:
        stat_sum.clear_cache()
        stat.clear_cache()
        for grid_batch_size in G:
            for f in fncs:
                _ = f(unifs, sim_batch_size, grid_batch_size)
                runtime = min(
                    timeit.repeat(
                        lambda: f(unifs, sim_batch_size, grid_batch_size),
                        repeat=2,
                        number=1,
                    )
                )
                d = dict(
                    n_arm_samples=n_arm_samples,
                    sim_batch_size=sim_batch_size,
                    grid_batch_size=grid_batch_size,
                    fnc=f.__name__,
                    time=runtime,
                )
                print(d)
                bench_data.append(d)


batching_benchmark(350)
```

```python
batching_benchmark(350, S=[2048, 4096], G=[1024])
```

```python
df = pd.DataFrame(bench_data)
df
```

```python
df.loc[df["time"].idxmin()]
```

## Is copying the uniforms array expensive?


```python
unifs = jax.random.uniform(jax.random.PRNGKey(0), (256000, 350, 4))
unifs10 = jax.random.uniform(jax.random.PRNGKey(0), (10000, 350, 4))
unifs2d = jax.random.uniform(jax.random.PRNGKey(0), (256000, 350, 1))
```

```python
for k in range(2):
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
        print("slice[:-1].sum", time.time() - start)

    start = time.time()
    out2 = unifs[0 : unifs.shape[0]].sum(axis=1).block_until_ready()
    if k >= 1:
        print("slice[0:n].sum", time.time() - start)

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

```python

```
