```python
from confirm.outlaw.nb_util import setup_nb
setup_nb()
```

```python
256000 * 350 * 4 * 8 / 1e9 / 9
```

```python
from confirm.models.binom1d import Binom1D
from confirm.imprint.cache import DuckDBCache
c = DuckDBCache.connect()
```

```python
%%time
m = Binom1D(0, 256000, n=350 * 4, cache=c)
```

```python
%%time
m = Binom1D(0, 256000, n=350 * 4, cache=c)
```

```python
import pandas as pd
import numpy as np
```

```python
%%time
b = m.samples.tobytes()
c.set("test2", pd.DataFrame(dict(data=[b])))
```

```python
import jax.numpy as jnp
```

```python
%%time
arr = jnp.frombuffer(c.get('test2')['data'].iloc[0], dtype=np.float32).reshape((256000, 1400))
```

```python
arr.shape
```

```python
arr.reshape((256000, 350 * 4)).shape
```

```python
import numpy as np
```

```python
import sys
import struct
def fast_numpy_save(array):
    size=len(array.shape)
    return bytes(array.dtype.byteorder.replace('=','<' if sys.byteorder == 'little' else '>')+array.dtype.kind,'utf-8')+array.dtype.itemsize.to_bytes(1,byteorder='little')+struct.pack(f'<B{size}I',size,*array.shape)+array.tobytes()

def fast_numpy_load(data):
    dtype = str(data[:2],'utf-8')
    dtype += str(data[2])
    size = data[3]
    shape = struct.unpack_from(f'<{size}I', data, 4)
    return np.ndarray(shape, dtype=dtype, buffer=data[4+size*4:])
```

```python
%%time
m = Binom1D(0, 256000, n=350 * 4, cache=c)
```

```python
c.get(c._get_all_keys().iloc[0,0])
```

```python
import numpy as np
A = np.random.uniform(0, 1, size=(3000, 3000))
```

```python
A
```

```python
from confirm.imprint.cache import DuckDBCache
c = DuckDBCache.connect()
c.set('abc', pd.DataFrame(dict(a=[1,2,3], b=[4,5,6])), shortname='cool')
```

```python
c.con.execute('select * from cache_tables').df()
```

```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()

import io

import numpy as np
import pandas as pd

import confirm.imprint as ip
```

```python
import json
json.dumps((0, {'a': (1,2)}))
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




class Binom1D:

    @staticmethod
    def unifs(seed, *, shape, dtype):
        return jax.random.uniform(jax.random.PRNGKey(seed), shape=shape, dtype=dtype)

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
        self.samples = cache(Binom1D.unifs)(seed, shape=(max_K, n), dtype=self.dtype)

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(self.samples[begin_sim:end_sim], theta, null_truth)
```

```python
class Cache:
    def __init__(self):
        self._cache = {}

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            key = json.dumps(dict(
                f=func.__module__ + '.' + func.__qualname__,
                args=args, 
                kwargs={str(k):str(v) for k,v in kwargs.items()}
            ))
            if key in self._cache:
                return self._cache[key]
            else:
                result = func(*args, **kwargs)
                self._cache[key] = result
                return result

        return wrapper
```

```python
import confirm.models.binom1d
c = Cache()
confirm.models.binom1d.Binom1D(c, 0, 100, n=100)
```

```python
from pprint import pprint
json.loads(list(c._cache.keys())[0])
```

```python
list(c._cache.keys())[0]
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

glob.glob("../../confirm/**/*.py")
```

```python
cache = Cache()
```

```python
%%time
unifs(0, shape=(10, 10), dtype=jnp.float32)
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
