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
p = jnp.array([0.05, 0.05])
p = jnp.zeros(2)
grid_points = jnp.array([p] * 1000)
n_sims = 1
key1 = jax.random.split(jax.random.PRNGKey(0), num=9)
keyN = jax.random.split(jax.random.PRNGKey(0), num=n_sims * 9).reshape((n_sims, 9, 2))
```

```python
print(len(jax.make_jaxpr(lei_obj.single_sim)(p, key1).pretty_print()))
print(len(jax.make_jaxpr(lei_obj.simulate_point)(p, keyN).pretty_print()))
print(len(jax.make_jaxpr(lei_obj.stage_1)(p, key1[:-1]).pretty_print()))
print(len(jax.make_jaxpr(lei_obj.posterior_sigma_sq)(np.random.rand(2,2)).pretty_print()))
# jax.make_jaxpr(lei_obj.posterior_sigma_sq)(np.random.rand(2,2))
```

```python
from numpyro.distributions.util import _binomial_dispatch
len(jax.make_jaxpr(_binomial_dispatch)(key1[0], 0.5, 30).pretty_print())
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
import jax
import numpyro.distributions as dist

def lets_break_this(p, key):
    n = (jax.random.uniform(key) > 0.5) * 3
    _, key = jax.random.split(key)
    return dist.Binomial(total_count=n, probs=p).sample(key)

print('this works fine')
jax.jit(lets_break_this)(0.1, jax.random.PRNGKey(0))
jax.jit(jax.vmap(lets_break_this, in_axes=(None, 0)))(
    0.1, jax.random.split(jax.random.PRNGKey(0), 1)
)

print('this works fine too!')
jax.jit(lets_break_this)(0, jax.random.PRNGKey(0))

print('jit with vmap and p=0 results in hanging forever')
jax.jit(jax.vmap(lets_break_this, in_axes=(None, 0)))(
    0, jax.random.split(jax.random.PRNGKey(0), 1)
)
```

```python

```
