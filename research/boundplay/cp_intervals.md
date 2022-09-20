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
import numpy as np
import jax.numpy as jnp
import jax
import scipy.stats
import matplotlib.pyplot as plt
```

## Investigating Clopper-Pearson




```python
%%time
n = 50
p = 0.2
thresh = 16
print('true type I error', 1 - scipy.stats.binom.cdf(thresh - 1, n, p))

niter = 1
ntests = 100000
nsims = 1000
delta = 0.01
test_ests = np.empty((niter, ntests))
n_failures = np.empty(niter, dtype=np.int32)
for i in range(niter):
    samples = scipy.stats.binom.rvs(n, p, size=nsims)
    reject = samples >= thresh
    typeI_sum = jnp.sum(reject)
    typeI_est = typeI_sum / nsims
    typeI_CI = scipy.stats.beta.ppf(1 - delta, typeI_sum + 1, nsims - typeI_sum) - typeI_est
    upper99 = typeI_est + typeI_CI

    test_ests[i, :] = jnp.mean(scipy.stats.binom.rvs(n, p, size=(ntests,nsims)) > thresh, axis=-1)
    n_failures[i] = jnp.sum(test_ests > upper99)
```

```python
empirical99 = np.percentile(test_ests.flatten(), 99)
```

```python
empirical99
```

```python
nsims = 40000000
samples = scipy.stats.binom.rvs(n, p, size=nsims)
reject = samples >= thresh
typeI_sum = jnp.sum(reject)
typeI_est = typeI_sum / nsims
typeI_CI = scipy.stats.beta.ppf(1 - delta, typeI_sum + 1, nsims - typeI_sum) - typeI_est
upper99_2 = typeI_est + typeI_CI
```

```python
typeI_est, typeI_CI
```

```python
plt.hist(test_ests.flatten())
plt.axvline(upper99, color='r', label='CP bound')
plt.axvline(upper99_2, color='r', label='CP bound 2')
plt.axvline(empirical99, color='k', label='empirical 99th percentile')
plt.xlabel('type I error')
plt.ylabel('count')
plt.legend()
plt.show()
```

```python
typeI_sum
```

```python
(n_failures.sum() / (niter * ntests))
```

## Check gradient bound

```python
n = 50
p = 0.2
t = jax.scipy.special.logit(p)
At = lambda t: -n * jnp.log(1 - jax.scipy.special.expit(t))
g = lambda t, x: t * x - At(t)
dg = jax.grad(g)
dg_vmap = jax.vmap(dg, in_axes=(None, 0))

holderp = 1.2
holderq = 6.0
# holderq = 1.0 / (1 - 1.0 / holderp)
```

```python
# numerical integral to compute E[ |grad g|^q]^(1/q)
# need to compute for the worst t in the "tile".
def C_numerical(t, p, holderq):
    xs = jnp.arange(n + 1).astype(jnp.float64)
    eggq = jnp.abs(dg_vmap(t, xs)) ** holderq
    return sum(eggq * scipy.stats.binom.pmf(xs, n, p)) ** (1 / holderq)

# Formula for C with q = 6 from wikipedia
def C_wiki(p):
    assert(holderq == 6)
    return (n * p * (1 - p) * (1 - 30 * p * (1 - p) * (1 - 4 * p * (1 - p)) + 5 * n * p * (1 - p) * (5 - 26 * p * (1 - p)) + 15 * n ** 2 * p ** 2 * (1 - p) ** 2)) ** (1 / 6)

# p = 0.2 corresponds to t=-1.386
# choose theta = -1.1 as the edge of our tile.
# so our tile is going to extend unidirectionally from -1.386 (the grid pt) to -1.1
# since the constant is monotonic in the relevant region, we can just compute
# at the edge of the tile to get a maximum value.
# tmax = -1.1
tmax = t
pmax = jax.scipy.special.expit(tmax)
C = C_wiki(pmax)
C_numerical(tmax, pmax, holderq), C_wiki(pmax)
```

```python
np.random.seed(0)
thresh = 20
nsims = 500000
print('true type I error', 1 - scipy.stats.binom.cdf(thresh - 1, n, p))
samples = scipy.stats.binom.rvs(n, p, size=nsims)
reject = samples >= thresh
typeI_sum = jnp.sum(reject)
typeI_est = typeI_sum / nsims
typeI_CI = scipy.stats.beta.ppf(1 - delta, typeI_sum + 1, nsims - typeI_sum) - typeI_est

grad_bound = []
for q in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    holderq = q
    holderp = 1.0 / (1 - 1.0 / holderq)
    print(holderq, holderp)
    C = C_numerical(tmax, pmax, holderq)
    grad_bound.append(C * (typeI_est + typeI_CI) ** (1 / holderp))
typeI_est + typeI_CI, grad_bound
```

```python
ntests = 100
new_samples = scipy.stats.binom.rvs(n, p, size=(ntests, nsims))
grad_est = np.mean((new_samples >= thresh) * (new_samples - n * p), axis=-1)
np.percentile(grad_est, 99), grad_bound
```

```python
min(grad_bound)
```

```python
np.where(grad_est > min(grad_bound))
```

```python
plt.hist(grad_est)
for i in range(len(grad_bound)):
    plt.axvline(grad_bound[i], color='r')
plt.show()
```

```python

```
