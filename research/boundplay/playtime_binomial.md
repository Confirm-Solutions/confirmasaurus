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
import matplotlib.pyplot as plt
import scipy.stats
from numpy import sqrt
import jax
import jax.numpy as jnp
```

```python
n = 50
p = 0.2
t = jax.scipy.special.logit(p)
At = lambda t: -n * jnp.log(1 - jax.scipy.special.expit(t))
g = lambda t, x: t * x - At(t)
dg = jax.grad(g)
dg_vmap = jax.vmap(dg, in_axes=(None, 0))
dgg_vmap = jax.vmap(jax.grad(dg), in_axes=(None, 0))

holderp = 1.2
holderq = 1.0 / (1 - 1.0 / holderp)
```

### The constant term

Compute the constant term using both a numerical sum and using the 6th moment formula from wikipedia

```python
# numerical integral to compute E[ |grad g|^q]^(1/q)
# need to compute for the worst t in the "tile".
def C_numerical(t, p):
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
tmax = -1.1
pmax = jax.scipy.special.expit(tmax)
# C = C_wiki(pmax)
C = C_numerical(tmax, pmax)
# C_numerical(tmax, pmax), C_wiki(pmax)
```

### Is the constant term monotonic? 

It's unimodal. Monotonic in [0, 0.5] and separately in [0.5, 1].

```python
ts = np.linspace(-10, 10, 100)
cs = [C_numerical(t, jax.scipy.special.expit(t)) for t in ts]
plt.plot(ts, cs)
plt.ylabel(r'$E[|\nabla g|^q]^{1/q}$')
plt.xlabel(r"$\theta$")
plt.show()
```

## Computing bounds for a simple binomial test.


For holder bounds, it's only necessary to choose a Type I Error value. 

But, to compute a classical gradient, it helps to have a real test that we're considering.

Here, we choose a test where we reject the null if x >= 19 and n=50. 

We're considering a tile where true p at the center of the tile is 0.2 and true theta is -1.38

The true Type I Error should be ~0.093%:

```python
thresh = 20
# the minus 1 is to account for cdf = p(x <= x) and we want p(x >= x)
1 - scipy.stats.binom.cdf(thresh - 1, n, p)
```

### Simulation

```python
delta = 0.01
nsims=int(1e4)
np.random.seed(0)
samples = scipy.stats.binom.rvs(n, p, size=nsims)
reject = samples >= thresh
typeI_sum = np.sum(reject)
typeI_est = typeI_sum / nsims
typeI_CI = scipy.stats.beta.ppf(1 - delta, typeI_sum + 1, nsims - typeI_sum) - typeI_est
typeI_est, typeI_CI
```

### Holder-ODE solution

```python
# use the upper bound on t1e
f0 = typeI_est + typeI_CI

# integration distance to reach the edge of the tile.
dt = tmax - t
t_path = np.linspace(t, t + dt, 100)

analytical = ((t_path - t) * C / holderq + f0 ** (1 / holderq)) ** holderq
```

### Classical bound

```python
delta_prop_0to1 = 0.5
typeI_CI_classic = scipy.stats.beta.ppf(1 - (delta * delta_prop_0to1), typeI_sum + 1, nsims - typeI_sum) - typeI_est

grad_est = np.sum(reject * (samples - n * p)) / nsims

covar = n * p * (1 - p)
grad_bound = np.sqrt(covar / nsims * (1 / ((1 - delta_prop_0to1) * delta) - 1))

# use pmax for a worst case hessian bound.
hess_bound = n * pmax * (1 - pmax)
```

Comparing gradient estimates: In this regime, classical gradient bound is ~10x larger than the Holder gradient bound. So, Holder gradient bound could be substituted in here for an improvement.

```python
grad_est + grad_bound, C * f0 ** (1 / holderp)
```

```python
classical = typeI_est + typeI_CI_classic + (grad_est + grad_bound) * (t_path - t) + 0.5 * hess_bound * (t_path - t) ** 2
```

### Second order Holder-ODI.


```python
f1 = min(C * f0 ** (1/holderp), grad_est + grad_bound)
holderp2 = 1.2
holderq2 = 1.0 / (1 - 1.0 / holderp2)
def C2_numerical(t, p):
    xs = jnp.arange(n + 1).astype(jnp.float64)
    integrand = jnp.abs(dg_vmap(t, xs) ** 2 + dgg_vmap(t, xs)) ** holderq2
    return sum(integrand * scipy.stats.binom.pmf(xs, n, p)) ** (1 / holderq2)
C2 = C2_numerical(tmax, pmax)
ts2 = np.linspace(-10, 10, 100)
cs2 = [C2_numerical(t, jax.scipy.special.expit(t)) for t in ts]
# plt.plot(ts2, cs2)
# plt.show()
def derivs2(_, y):
    cur_f = y[0]
    fp = y[1]
    fpp = C2 * cur_f ** (1 / holderp2)
    return [fp, fpp]
result2 = scipy.integrate.solve_ivp(derivs2, (t, t+dt), [f0, f1], t_eval=t_path, rtol=1e-10, atol=1e-10)
holderode2 = result2['y'][0]
# holderode2
```

## Comparing bounds

```python
plt.plot([t], [f0], 'ko')
plt.plot(t_path, analytical, 'b-', label='holder-ode')
plt.plot(t_path, classical, 'r--', label='classical')
# plt.plot(t_path, holderode2, 'k--', label='holder-ode2')
plt.xlabel(r'$\theta$')
plt.ylabel('type I error')
plt.legend()
plt.show()
```

```python
1 - scipy.stats.binom.cdf(thresh - 1, n, scipy.special.expit(-1.0))
```

Computing the bound at the upper edge of the tile.

```python
analytical[-1], classical[-1]
```

Computing the point where the bound crosses 0.025 and then compute the maximum cell size that would avoid the bound crossing 0.025.

```python
holder_idx = np.argmin(analytical < 0.025)
classical_idx = np.argmin(classical < 0.025)
holder_cell_size = (t_path[holder_idx] - t)
classical_cell_size = (t_path[classical_idx] - t)

t_path[holder_idx], t_path[classical_idx], holder_cell_size, classical_cell_size
```

The ratio of cell sizes. This is a guess at the ratio of the number of cells that would be required if we switched to using holder in regions of comparably low T1E. This corresponds to a reduction of 500x in the number of tiles required for a 4D problem with low T1E.

```python
holder_cell_size / classical_cell_size
```

## Checking the ODE integrator

Against the analytical solution

```python
def derivs(_, y):
    cur_f = y[0]
    return C * cur_f ** (1 / holderp)
holderode = scipy.integrate.solve_ivp(derivs, (t, t+dt), [f0], t_eval=t_path, rtol=1e-10, atol=1e-10)
```

```python
error = analytical - holderode['y'][0,:]
plt.plot(t_path, np.log10(np.abs(error)))
plt.title('absolute error in holder-ode')
plt.xlabel(r'$\theta$')
plt.ylabel('$\log_{10}(|f_N - f_A|)$')
plt.show()
```

```python
plt.plot(t_path, np.log10(np.abs(error / analytical)))
plt.title('relative error in holder-ode')
plt.xlabel(r'$\theta$')
plt.ylabel('$\log_{10}(|(f_N - f_A) / f_A|)$')
plt.show()
```
