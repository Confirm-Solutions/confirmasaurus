---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('imprint')
    language: python
    name: python3
---

```python
import inlaw
import inlaw.berry as berry
import inlaw.quad as quad
import numpy as np
import jax.numpy as jnp
import jax
import time
import inlaw.inla as inla
```

```python
def my_timeit(N, f, iter=5, inner_iter=10, should_print=True):
    _ = f()
    runtimes = []
    for i in range(iter):
        start = time.time()
        f()
        runtimes.append(time.time() - start)
    if should_print:
        print("median runtime", np.median(runtimes))
        print("min us per sample ", np.min(runtimes) * 1e6 / N)
        print("median us per sample", np.median(runtimes) * 1e6 / N)
    return runtimes

def benchmark(N=10000, iter=5):
    dtype = np.float32
    data = berry.figure2_data(N).astype(dtype)
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts.astype(dtype)
    x0 = jnp.zeros((sig2.shape[0], 4), dtype=dtype)

    print("\ncustom dirty bayes")
    db = jax.jit(jax.vmap(berry.build_dirty_bayes(sig2, n_arms=4, dtype=dtype)))
    my_timeit(N, lambda: db(data)[0].block_until_ready(), iter=iter)

    print("\ncustom dirty bayes")
    db = jax.jit(jax.vmap(berry.build_dirty_bayes(sig2, n_arms=4, dtype=dtype)))
    my_timeit(N, lambda: db(data)[0].block_until_ready(), iter=iter)

    def bench_ops(name, ops):
        print(f"\n{name} gaussian")
        hyperpost = jax.jit(jax.vmap(ops.laplace_logpost, in_axes=(None, None, 0)))
        p_pinned = dict(sig2=sig2, theta=None)
        my_timeit(
            N, lambda: hyperpost(x0, p_pinned, data)[0].block_until_ready(), iter=iter
        )

        print(f"\n{name} laplace")
        _, x_max, hess_info, _ = hyperpost(x0, p_pinned, data)
        arm_logpost_f = jax.jit(
            jax.vmap(
                jax.vmap(
                    ops.cond_laplace_logpost, in_axes=(0, 0, None, 0, 0, None, None)
                ),
                in_axes=(None, None, None, None, 0, None, None),
            ),
            static_argnums=(5, 6),
        )
        invv = jax.jit(jax.vmap(jax.vmap(ops.invert)))

        def f():
            inv_hess = invv(hess_info)
            arm_post = []
            for arm_idx in range(4):
                cx, wts = inla.gauss_hermite_grid(
                    x_max, inv_hess[..., arm_idx, :], arm_idx, n=25
                )
                arm_logpost = arm_logpost_f(
                    x_max, inv_hess[:, :, arm_idx], p_pinned, data, cx, arm_idx, True
                )
                arm_post.append(inla.exp_and_normalize(arm_logpost, wts, axis=0))
            return jnp.array(arm_post)

        my_timeit(N, jax.jit(f), iter=iter)

    custom_ops = berry.optimized(sig2, dtype=dtype).config(max_iter=10)
    bench_ops("custom berry", custom_ops)

    ad_ops = inla.from_log_joint(
        berry.log_joint(4), dict(sig2=np.array([np.nan]), theta=np.full(4, 0.0))
    ).config(max_iter=10)
    bench_ops("numpyro berry", ad_ops)
```

```python
N = 1
dtype = np.float64
data = berry.figure2_data(N).astype(dtype)
sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
sig2 = sig2_rule.pts.astype(dtype)
x0 = jnp.zeros((sig2.shape[0], 4), dtype=dtype)

ad_ops = inla.from_log_joint(
    berry.log_joint(4), dict(sig2=np.array([np.nan]), theta=np.full(4, 0.0))
).config(max_iter=10)

hyperpost = jax.jit(jax.vmap(ad_ops.laplace_logpost, in_axes=(None, None, 0)))
p_pinned = dict(sig2=sig2, theta=None)
out = hyperpost(x0, p_pinned, data)
```

```python
out
```
