---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.6 ('confirm')
    language: python
    name: python3
---

# Tight Bounds for Binomial

```python
%load_ext autoreload
%autoreload 2
```

```python
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import cvxpy as cp
import confirm.mini_imprint.bound.binomial as binomial
```

This notebook studies the behavior of a collection of bounds in a simple binomial test setting.
Consider $X \sim Binom(n, p(\theta))$ where $n$ is fixed, $\theta \in \mathbb{R}$ is the natural parameter,
and $p(\theta)$ is the sigmoid function.
For a fixed critical threshold $t^*$, we reject if $X > t^*$.


## Taylor Bound

```python
def f(theta, n, t):
    return scipy.stats.binom.sf(t, n, scipy.special.expit(theta))
    
def df(theta, n, t):
    p = scipy.special.expit(theta)
    return scipy.stats.binom.expect(
        lambda x: (x > t) * (x - n*p),
        args=(n, p),
    )
```

```python
def taylor_bound(f0, df0, vs, theta_0, n):
    p = scipy.special.expit(theta_0)
    return f0 + df0 * vs + 0.5 * vs**2 * n * p * (1-p)
```

## Centered Holder Bound

```python
def copt(a, p):
    return 1 / (1 + ((1-a)/a)**(1/(p-1)))

def C_numerical(n_arm_samples, t, hp, hq):
    p = scipy.special.expit(t)
    xs = np.arange(n_arm_samples + 1).astype(np.float64)
    eggq = np.abs(xs - n_arm_samples * p[:, None]) ** hq
    return np.sum(eggq * scipy.stats.binom.pmf(xs, n_arm_samples, p[:, None]), axis=-1) ** (1 / hq)
    
def holder_bound(f0, n_arm_samples, theta_0, vs, hp, hc='opt'):
    if isinstance(hp, np.ndarray):
        bounds = np.array([holder_bound(f0, n_arm_samples, theta_0, vs, hpi, hc) for hpi in hp])
        return np.min(bounds, axis=0)
    if hc == 'opt':
        hc = copt(f0, hp)
    hq = 1 / (1 - 1 / hp)
    B = hc ** hp
    A = (1 - hc) ** hp - B
    Cs = [
        scipy.integrate.quadrature(
            lambda h: np.abs(v) * C_numerical(n_arm_samples, theta_0 + h * v, hp, hq),
            0.0,
            1.0,
        )[0]
        for v in vs
    ]
    Cs = np.maximum.accumulate(Cs)
    return 1/A * (A*Cs / hq + (A*f0 + B)**(1/hq))**hq - B/A
```

## Exponential Holder Improved

```python
def log_partition(t, n):
    return n * jnp.log(1 + jnp.exp(t))
```

```python
def exp_holder_impr_bound_(f0, n, theta_0, vs, q):
    A0 = log_partition(theta_0, n)
    bounds = f0**(1-1/q) * np.exp(
        (log_partition(theta_0 + q * vs, n) - A0) / q
        - (log_partition(theta_0 + vs, n) - A0)
    )
    return bounds

def exp_holder_impr_bound(f0, n, theta_0, vs, q = 'inf'):
    if isinstance(q, np.ndarray):
        bounds = np.array([exp_holder_impr_bound_(f0, n, theta_0, vs, qi)[1] for qi in q])
        order = np.argmin(bounds, axis=0)
        return q[order], bounds[order, np.arange(0, len(order))]
    elif q == 'inf' or (isinstance(q, float) and np.isinf(q)): 
        return None, f0 * np.exp(n*vs - log_partition(theta_0 + vs, n) + log_partition(theta_0, n))
    elif q == 'opt':
        solver = binomial.ForwardQCPSolver(n, qcp_convg_tol=1e-4)
        q_solver = jax.jit(jax.vmap(solver.solve, in_axes=(None, 0, None)))
        qs = q_solver(theta_0, vs, f0)
        bounds_f = jax.vmap(binomial.q_holder_bound_fwd, in_axes=(0, None, None, 0, None))
        bounds = bounds_f(qs, n, theta_0, vs, f0)
        return qs, bounds
    bounds = exp_holder_impr_bound_(f0, n, theta_0, vs, q)
    return None, bounds
```

## Performance Comparison

```python
n = 350
theta_0 = -0.5
theta_boundary = 0
v_max = theta_boundary - theta_0
n_steps = 100
alpha = 0.025
p_boundary = scipy.special.expit(theta_boundary)
thresh = np.sqrt(n*p_boundary*(1-p_boundary)) * scipy.stats.norm.isf(alpha) + n*p_boundary
```

```python
f0 = f(theta_0, n, thresh)
df0 = df(theta_0, n, thresh)
vs = np.linspace(1e-8, v_max, n_steps)
```

```python
def run(theta_0, n, f0, df0, vs, thresh, hp, hc, q='inf'):
    # compute true Type I Error
    thetas = theta_0 + vs
    fs = f(thetas, n, thresh)

    # compute taylor bound
    taylor_bounds = taylor_bound(f0, df0, vs, theta_0, n)

    # compute holder centered bound
    holder_bounds = [holder_bound(f0, n, theta_0, vs, hp, c) for c in hc]
    
    # compute exp holder impr bound
    qs, exp_holder_impr_bounds = exp_holder_impr_bound(f0, n, theta_0, vs, q)

    # plot everything
    plt.plot(thetas, fs, ls='--', color='black', label='True TIE')
    plt.plot(thetas, taylor_bounds, ls='-', label='taylor')
    for i, c in enumerate(hc):
        plt.plot(thetas, holder_bounds[i], ls='--', label=f'centered-holder({c}), p={hp}')
    plt.plot(thetas, exp_holder_impr_bounds, ls=':', label='exp-holder-impr')
    plt.ylim(np.maximum(np.min(fs)-1e-7, 0), np.max(exp_holder_impr_bounds)+1e-7)
    plt.legend()
    plt.show()
    
    return qs
```

```python
qs = run(
    theta_0=theta_0,
    n=n,
    f0=f0,
    df0=df0,
    vs=vs,
    thresh=thresh,
    hp=1.1,
    hc=['opt'],
    q='opt',
)
```

```python
solver = binomial.ForwardQCPSolver(n)
q_opt = solver.solve(theta_0, vs[-1], f0)
qs_plt = np.linspace(1.00001, 100, 1000)
objs = [solver.objective(q, theta_0, vs[-1], f0) for q in qs_plt]
plt.plot(qs_plt, objs)
plt.plot(q_opt, solver.objective(q_opt, theta_0, vs[-1], f0), 'r.')
```

```python
np.min(objs), solver.objective(q_opt, theta_0, vs[-1], f0)
```
