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
import jax
import jax.numpy as jnp
import numpy as np

def memory_status(title):
    client = jax.lib.xla_bridge.get_backend()
    mem_usage = sum([b.nbytes for b in client.live_buffers()]) / 1e9
    print(f'{title} memory usage', mem_usage)
    print(f'{title} buffer sizes', [b.shape for b in client.live_buffers()])
    
key1 = jax.random.PRNGKey(0)
unifs = jax.random.uniform(key=key1, shape=(256000, 350, 4), dtype=jnp.float32)
def f(x):
    return jnp.sum(x, axis=1)
fj = jax.jit(f)
for i in range(2):
    for size in [1000,2000,4000,8000,16000,32000,64000,128000,256000]:
        subset = unifs[:size]
        fv = np.empty((size, unifs.shape[2]))
        for i in range(size//1000):
            fv[i*1000:(i+1)*1000] = fj(subset[i*1000:(i+1)*1000])
        print(size, fv.shape)
        memory_status('report')
```

```python
print('hi')
```
