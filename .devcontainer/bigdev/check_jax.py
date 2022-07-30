import time

import jax
import numpy as np

x = np.random.rand(5000, 5000)
start = time.time()
x.dot(x)
print("numpy", time.time() - start)

for i in range(10):
    xx = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), shape=(5000, 5000)))
    start = time.time()
    # yy = jax.device_get(xx.dot(xx))
    xx.dot(xx).block_until_ready()
    print("jax", i, time.time() - start)
