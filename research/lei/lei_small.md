---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('confirm')
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import jax
import jax.numpy as jnp
import numpy as np

from lei_obj import Lewis45
```

```python
params = {
    "n_arms" : 2,
    "n_stage_1" : 50,
    "n_interims" : 1,
    "n_add_per_interim" : 100,
    "futility_threshold" : 0.1,
    "n_stage_2" : 100,
    "pps_threshold_lower" : 0.1,
    "pps_threshold_upper" : 0.9,
    "posterior_difference_threshold" : 0.1,
    "rejection_threshold" : 0.05,
}

lei_obj = Lewis45(**params)
p = jnp.zeros(2)
grid_points = jnp.array([p] * 1000)
n_sims = 1
key1 = jax.random.split(jax.random.PRNGKey(0), num=4)
keyN = jax.random.split(jax.random.PRNGKey(0), num=n_sims * 4).reshape((n_sims, 4, 2))
```

```python
print(len(jax.make_jaxpr(lei_obj.single_sim)(p, key1).pretty_print()))
print(len(jax.make_jaxpr(lei_obj.simulate_point)(p, keyN).pretty_print()))
print(len(jax.make_jaxpr(lei_obj.stage_1)(p, key1[:-1]).pretty_print()))
print(len(jax.make_jaxpr(lei_obj.posterior_sigma_sq)(np.random.rand(2,2)).pretty_print()))
# jax.make_jaxpr(lei_obj.posterior_sigma_sq)(np.random.rand(2,2))
```

```python
%%time
# lei_obj.single_sim(p, keyN[0])
# rejections = jax.jit(lei_obj.single_sim)(p, key1)
# jax.vmap(lei_obj.single_sim, in_axes=(None, 0))(p, keyN)
rejections = jax.jit(lei_obj.simulate_point)(p, keyN)
#rejections = jax.jit(lei_obj.simulate, static_argnums=(0, 3))(n_sims, grid_points, key, 1)
```

```python

```
