---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.10.5
    language: python
    name: python3
---

<!-- #region -->
## Numerical play with bounds.

For everything in this notebook, refer back to the google doc for context. https://docs.google.com/document/d/1w2w41N5nk3-Nz9Pqfw4B1Gty_xzKy6Z-MP2bwbu0YuY/edit


Some notes
- concentration inequalities: https://en.wikipedia.org/wiki/Concentration_inequality
- Vysochanskij–Petunin_inequality: 
    - https://www.johndcook.com/blog/2016/02/12/improving-on-chebyshevs-inequality/
- learning about exponential families and sufficient statistics: https://eml.berkeley.edu/~mcfadden/e240a_sp01/sufficiency.pdf
- discrete unimodality and strong unimodality? https://www.jstor.org/stable/2283941
- strong unimodality? https://epubs.siam.org/doi/epdf/10.1137/1101021
- sum of unimodal RVs: https://math.stackexchange.com/questions/70651/is-the-sum-of-independent-unimodal-random-variables-still-unimodal


What are the different types of gradient bounds that we have available to us?
- Ignoring summation, treating the estimate as a black box with a bound on the variance of the estimate: Cantelli/Chebyshev-type concentration inequalities (e.g. Vysochanskij)
- Continuous Cauchy-Schwartz/Holder-type bounds. (ignoring summation except insofar as the Clopper-Pearson bound for the error treats it)
- Treating the summation directly via a version of the central limit theorem?
- Bounded variables can use Hoeffding or Chernoff. But we don't have bounded variables.

Quantities that are easy to get:
- unbiased estimate of T1E and its gradient
- analytical or numerical evaluation of E[d^n (theta T - A)]
<!-- #endregion -->

## Setup

This is just a duplication of the code from the `normal_seq.ipynb` notebook.

```python
import confirm.outlaw.nb_util as nb_util

nb_util.setup_nb()
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from numpy import sqrt
import jax
import jax.numpy as jnp

delta = 0.01
z_thresh = 1.96
npts = 2
a = -1
npts = 100
a = -2
b = 0
np.random.seed(9)

mu = np.linspace(a, b, 2 * npts + 1)[1::2]
stepsize = mu[1] - mu[0]

z_thresh = -jax.scipy.stats.norm.ppf(0.025)
true_err = lambda mu: 1 - jax.scipy.stats.norm.cdf(-mu + z_thresh)

# these gradients are equivalent
true_gradient = jax.vmap(jax.grad(true_err))
true_gradient2 = lambda mu: jax.scipy.stats.norm.pdf(-mu + z_thresh)
true_second = jax.vmap(jax.grad(jax.grad(true_err)))

# simulating the rejection/type I error
nsims = 100000
samples = np.random.normal(
    mu[:, None],
    1,
    size=(
        mu.shape[0],
        nsims,
    ),
)
reject = samples > z_thresh
typeI_sum = np.sum(reject, axis=-1)
typeI_est = typeI_sum / nsims

grad_est = np.sum(reject * (samples - mu[:, None]), axis=-1) / nsims

typeI_CI = scipy.stats.beta.ppf(1 - delta, typeI_sum + 1, nsims - typeI_sum) - typeI_est
chebyshev = np.sqrt(1 / (delta * nsims))
cantelli = np.sqrt(1 / nsims * (1 / delta - 1))

# the actual uniform upper bound on the second order term is ~0.2419707
# and occurs at N(x = 1, mu = 0, sig = 1) because x * np.exp(-0.5 * x ** 2) is
# maximized at x = 1
hess_bound_true = -scipy.optimize.minimize(lambda x: -true_second(x), 0).fun

# we can check the variance explicitly. note that the variance won't change
# with different values of mu because the whole integrand is just translated by
# mu.
explicit_integral = scipy.integrate.quad(
    lambda x: scipy.stats.norm.pdf(x, 0) * (x - 0) ** 2, -10, 10
)
hess_bound = explicit_integral[0]
```

```python
# Demonstration that the gradient estimate is very very close to normally distributed. CLT and all that.
more_samples = np.random.normal(
    mu[-1, None],
    1,
    size=(
        1,
        nsims,
    ),
)
plt.hist(
    np.mean(more_samples[-1, :].reshape((-1, 5)), axis=-1) - mu[-1],
    bins=100,
    density=True,
)
plt.show()
```

## Holder bounds

```python
# egg2 = expectation of gradient of g squared.
egg2_uniform = 1.0
egg2_empirical = np.sum(reject * (samples - mu[:, None]) ** 2, axis=-1) / nsims

egg2_f = lambda mu: scipy.integrate.quad(
    lambda x: scipy.stats.norm.pdf(x, mu) * (x - mu) ** 2, z_thresh, 10
)[0]
egg2_true = np.array([egg2_f(m) for m in mu])
```

```python
# def egQ(q):
#     I = scipy.integrate.quad(
#         lambda x: scipy.stats.norm.pdf(x, 0) * np.abs(x) ** q, -10, 10
#     )
#     print(I[1])
#     return I[0]

# For small p, q is large and we should use more precision
import mpmath as mp

mp.mp.dps = 100
gaussian_pdf = lambda x: (1 / mp.sqrt(2 * mp.pi)) * mp.exp(-(x**2) / 2)


def egQ(q):
    return float(
        mp.re(
            mp.quad(lambda x: gaussian_pdf(x) * np.abs(x) ** q, [-10, 10], error=True)[
                0
            ]
        )
    )


def qf(p):
    return 1 / (1 - 1 / p)


def holder_bound(p):
    q = qf(p)
    return ((typeI_est + typeI_CI) ** (1 / p)) * (egQ(q) ** (1 / q))


holder_bound(4)
```

```python
egQ(1.5) ** (1 / 1.5)
```

```python
egQ(1.01)
```

```python
egQ(3.0)
```

```python
style = ["b-", "r--", "g-.", "k:"]
for i, p in enumerate([1.2, 1.6, 3]):
    print(qf(p))
    plt.plot(mu, holder_bound(p), style[i], label=f"holder p={p}")
plt.plot(mu, np.sqrt((typeI_est + typeI_CI) * egg2_uniform), "k-", label="holder p=2")
plt.plot(mu[::5], np.full_like(mu, cantelli)[::5], "ko", label="cantelli")
plt.xlabel("$\mu$")
plt.ylabel("gradient upper bound")
plt.legend()
plt.show()
```

```python
[egQ(qf(p)) ** (1 / qf(p)) for p in [1.023, 1.054, 1.09, 1.2]]
```

```python
qf(1.2)
```

Going below p=1.2 doesn't help for this problem.

```python
for i, p in enumerate([1.023, 1.054, 1.09, 1.2]):
    plt.plot(mu, holder_bound(p), style[i], label=f"holder p={p}")
plt.xlabel("$\mu$")
plt.ylabel("gradient upper bound")
plt.legend()
plt.show()
```

```python
grad_f = lambda mu: scipy.integrate.quad(
    lambda x: scipy.stats.norm.pdf(x, mu) * (x - mu), z_thresh, 10
)[0]
grad_true = np.array([grad_f(m) for m in mu])

plt.plot(mu, grad_true, label="true")
plt.plot(mu, grad_est, label="empirical")
plt.fill_between(mu, grad_true, grad_true + holder_bound(1.2), alpha=0.5)
plt.show()
```

```python
C = scipy.integrate.quad(
    lambda x: scipy.stats.norm.pdf(x, 0) * (x**2 + 1) ** 2, -10, 10
)[0]
C
```

## Solving the hessian ivp that Mike posed...

```python
C = 6.0
mu = -0.25
dmu = 0.125
f0 = float(true_err(np.array([mu])))
g0 = float(true_gradient(np.array([mu])))


def f(t, y):
    return np.array([y[1], C * np.sqrt(y[0])])


mu_path = np.linspace(mu, mu + dmu, 100)
solution = scipy.integrate.solve_ivp(f, (mu, mu + dmu), [f0, g0], t_eval=mu_path)
plt.plot(mu_path, solution["y"][0, :], "b-")
plt.show()
```

## Cauchy-Schwartz on the third derivative

```python
g = lambda t, x: t * x - t**2 / 2
dg = jax.grad(g)
ddg = jax.grad(jax.grad(g))
dddg = jax.grad(jax.grad(jax.grad(g)))
t = 0.0
C = scipy.integrate.quad(
    lambda x: scipy.stats.norm.pdf(x, 0) * (-3 * x + x**3) ** 2, -10, 10
)[0]
C
```

```python
n = 150
p = 0.5
holderp = 3.0
holderq = qf(holderp)
At = lambda t: -n * jnp.log(
    1 - jax.scipy.special.expit(t)
)  # (t + jnp.log(1 + jnp.exp(-t)))
g = lambda t, x: t * x - At(t)
dg = jax.grad(g)
ddg = jax.grad(jax.grad(g))
dddg = jax.grad(jax.grad(jax.grad(g)))
t = scipy.special.logit(p)
integrand = lambda x, q: np.abs(
    3 * dg(t, x) * ddg(t, x) + dddg(t, x) + dg(t, x) ** 3
) ** q * scipy.stats.binom.pmf(x, n, p)
ugly = (sum([integrand(i, holderq) for i in range(n + 1)])) ** (1 / holderq)
ugly
```

```python
holderq
```

```python
ugly * (0.005) ** (1 / holderp)
```

```python
n * (1 - p) * p
```

```python
37.5 * 0.05**2 / 2, 72.5 * 0.05**3 / 6
```

## scrap

```python
# sigR2 = scipy.integrate.quad(lambda x: scipy.stats.norm.pdf(x, 0) * x ** 2, -10, 10)[0]
# rho = scipy.integrate.quad(lambda x: scipy.stats.norm.pdf(x, 0) * np.abs(x ** 3), -10, 10)[0]
# bound = 0.5 * rho / (sigR2 ** 3 * np.sqrt(nsims))
# bound
```
