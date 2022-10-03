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
import confirm.outlaw.nb_util as nb_util
nb_util.setup_nb()
```

```python
import jax
import os
import numpy as np
from confirm.mini_imprint import grid
from confirm.lewislib import grid as lewgrid
from confirm.lewislib import lewis
from confirm.mini_imprint import binomial
```

```python
# Configuration used during simulation
params = {
    "n_arms" : 4,
    "n_stage_1" : 50,
    "n_stage_2" : 100,
    "n_stage_1_interims" : 2,
    "n_stage_1_add_per_interim" : 100,
    "n_stage_2_add_per_interim" : 100,
    "stage_1_futility_threshold" : 0.15,
    "stage_1_efficacy_threshold" : 0.7,
    "stage_2_futility_threshold" : 0.2,
    "stage_2_efficacy_threshold" : 0.95,
    "inter_stage_futility_threshold" : 0.6,
    "posterior_difference_threshold" : 0,
    "rejection_threshold" : 0.05,
    "key" : jax.random.PRNGKey(0),
    "n_pr_sims" : 100,
    "n_sig2_sims" : 20,
    "batch_size" : int(2**20),
    "cache_tables" : './lei_cache.pkl',
}

params = {
    "n_arms" : 4,
    "n_stage_1" : 15,
    "n_stage_2" : 33,
    "n_stage_1_interims" : 2,
    "n_stage_1_add_per_interim" : 33,
    "n_stage_2_add_per_interim" : 33,
    "stage_1_futility_threshold" : 0.15,
    "stage_1_efficacy_threshold" : 0.7,
    "stage_2_futility_threshold" : 0.2,
    "stage_2_efficacy_threshold" : 0.95,
    "inter_stage_futility_threshold" : 0.6,
    "posterior_difference_threshold" : 0,
    "rejection_threshold" : 0.05,
    "key" : jax.random.PRNGKey(0),
    "n_pr_sims" : 100,
    "n_sig2_sims" : 20,
    "batch_size" : int(2**20),
    "cache_tables" : './lei_cache.pkl',
}
size = 52
n_sim_batches = 500
sim_batch_size = 100
```

```python
# construct Lei object
lei_obj = lewis.Lewis45(**params)
```

```python

```
