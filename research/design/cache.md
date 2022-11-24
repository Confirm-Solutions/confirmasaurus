```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()

import io

import numpy as np
import pandas as pd

import confirm.imprint as ip
```

```python
import jax
import jax.numpy as jnp


@jax.jit
def _sim(samples, theta, null_truth):
    p = jax.scipy.special.expit(theta)
    stats = jnp.sum(samples[None, :] < p[:, None], axis=2) / samples.shape[1]
    return jnp.where(
        null_truth[:, None, 0],
        1 - stats,
        jnp.inf,
    )


def unifs(seed, *, shape, dtype):
    return jax.random.uniform(jax.random.PRNGKey(seed), shape=shape, dtype=dtype)


class Binom1D:
    def __init__(self, cache, seed, max_K, *, n):
        self.family = "binomial"
        self.family_params = {"n": n}
        self.dtype = jnp.float32

        # cache_key = f'samples-{seed}-{max_K}-{n}-{self.dtype}'
        # if cache_key in cache:
        #     self.samples = cache[cache_key]
        # else:
        #     key = jax.random.PRNGKey(seed)
        #     self.samples = jax.random.uniform(key, shape=(max_K, n), dtype=self.dtype)
        #     cache.update({cache_key: self.samples})
        #
        self.samples = cache(unifs)(seed, shape=(max_K, n), dtype=self.dtype)

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
```

```python
class Cache:
    def __init__(self):
        self._cache = {}

    def __call__(self, func, safety=2, serialize=False):
        def wrapper(*args, **kwargs):
            key = (func, args, tuple(kwargs.items()))
            if key in self._cache:
                return self._cache[key]
            else:
                result = func(*args, **kwargs)
                self._cache[key] = result
                return result

        return wrapper
```

```python
import os
import hashlib
import confirm


def hash_confirm_code():
    confirm_path = os.path.dirname(confirm.__file__)
    hashes = []
    hash_md5 = hashlib.md5()
    for path, subdirs, files in os.walk(confirm_path):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(path, fn), "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    hash_md5.hexdigest()
```

```python
import glob

glob.glob("confirm/*.py")
```

```python
cache = Cache()
```

```python
%%time
unifs(0, shape=(10, 10), dtype=jnp.float32)
```

```python
t.results().write_results(coverdir=".")
```

```python
%%timeit
model = Binom1D(cache, 0, 1000000, n=100)
```

```python
cache._cache
```

## old

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
rej_df = ip.validate(Binom1D, g, 0.5, K=2**10, model_kwargs={"n": 100})
rej_df
```

```python
model = Binom1D(0, 2**18, n=100)
db = ip.db.DuckDB.connect()
```

```python
%%time
samples = pd.DataFrame(model.samples)
```

```python
%%timeit
memfile = io.BytesIO()
np.save(memfile, samples.values)
```

```python
%%timeit
memfile.seek(0)
s2 = np.load(memfile)
```

```python
np.all(s2 == samples.values)
```

```python
%%timeit
db.con.execute("drop table samples")
db.con.execute("create table samples as select * from samples")
```
