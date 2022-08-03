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
import jax.numpy as jnp
import numpy as np
import jax
from lei_obj import Lewis45
```

```python
params = {
    "n_arms" : 2,
    "n_stage_1" : 50,
    "n_interims" : 3,
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
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, num=n_sims)
```

```python
compiled = jax.jit(lei_obj.single_sim).lower(p, key).compile()
```

```python
%%time
compiled(p, key)
```

```python
%%time
rejections = jax.jit(lei_obj.simulate_point)(p, keys)
#rejections = jax.jit(lei_obj.simulate, static_argnums=(0, 3))(n_sims, grid_points, key, 1)
```

```python

```
