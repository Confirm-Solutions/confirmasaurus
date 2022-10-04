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
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import cvxpy as cp
```

This notebook studies the behavior of a collection of bounds in a simple binomial test setting.
Consider $X \sim Binom(n, p(\theta))$ where $n$ is fixed, $\theta \in \mathbb{R}$ is the natural parameter,
and $p(\theta)$ is the sigmoid function.
For a fixed critical threshold $t^*$, we reject if $X > t^*$.


\begin{align*}
A(\theta) &:= n\log(1 + e^{\theta})\\
A(\theta_0 + qv) - A(\theta_0)
&:=
n\left(\log(1 + e^{\theta_0+qv}) - \log(1 + e^{\theta_0})\right)
\\&\leq
n q|v|
\end{align*}


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

def opt_q(theta_0, n, v, a, eps=1e-4, max_iters=1000):
    def f(u, v, a):
        q = jnp.exp(u) + 1
        return (log_partition(theta_0 + q*v, n) - log_partition(theta_0, n) + a) / q 
    def df(u, v, a):
        # up to a factor of 1/q^2
        q = jnp.exp(u) + 1
        t = theta_0 + q * v
        return v * jax.scipy.special.expit(t) * q \
            - (jnp.logaddexp(0, t) - jnp.logaddexp(0, theta_0)) - a / n
    ddf = jax.grad(lambda u: df(u, v, a))
    u_opt = 0.0
    #f_opt_prev = np.inf
    #f_opt = f(u_opt, v, a)
    df_u_opt = df(u_opt, v, a)
    i = 0
    while np.abs(df_u_opt) >= eps and i < max_iters:
        ddf_u_opt = ddf(u_opt) 
        sgn_df = np.sign(df_u_opt)
        sgn_ddf_u_opt = np.sign(ddf_u_opt)
        if ddf_u_opt == 0:
            u_opt = -sgn_df * np.inf
            break
        if sgn_df != sgn_ddf_u_opt and \
            np.log(np.abs(df_u_opt)) - np.log(np.abs(ddf_u_opt)) < np.log(1e6):
            u_opt = np.inf
            break
        u_opt -= df_u_opt / ddf_u_opt
        df_u_opt = df(u_opt, v, a)
        #f_opt_prev = f_opt
        #f_opt = f(u_opt, v, a)
        i += 1
    return jnp.exp(u_opt) + 1
```

```python
def exp_holder_impr_bound(f0, n, theta_0, vs, q = 'inf'):
    if isinstance(q, np.ndarray):
        bounds = np.array([exp_holder_impr_bound(f0, n, theta_0, vs, qi)[1] for qi in q])
        order = np.argmin(bounds, axis=0)
        return q[order], bounds[order, np.arange(0, len(order))]
    if q == 'inf' or (isinstance(q, float) and np.isinf(q)): 
        return None, f0 * np.exp(n*vs - log_partition(theta_0 + vs, n) + log_partition(theta_0, n))
    if q == 'opt':
        a = -np.log(f0)
        q = np.array([opt_q(theta_0, n, v, a) for v in vs])
    A0 = log_partition(theta_0, n)
    return None, f0 * np.exp(
        (log_partition(theta_0 + q * vs, n) - A0) / q
        - np.log(f0) / q
        - (log_partition(theta_0 + vs, n) - A0)
    )
```

## Performance Comparison

```python
n = 500
theta_0 = -1
v_max = 0.1
n_steps = 100
alpha = 0.1
p0 = scipy.special.expit(theta_0)
thresh = np.sqrt(n*p0*(1-p0)) * scipy.stats.norm.isf(alpha) + n*p0
```

```python
f0 = f(theta_0, n, thresh)
df0 = df(theta_0, n, thresh)
vs = np.linspace(0, v_max, n_steps)
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
    #plt.plot(thetas, taylor_bounds, ls='-', label='taylor')
    for i, c in enumerate(hc):
        plt.plot(thetas, holder_bounds[i], ls='--', label=f'centered-holder({c}), p={hp}')
    plt.plot(thetas, exp_holder_impr_bounds, ls=':', label='exp-holder-impr')
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
    hp=np.logspace(0.1, 5, 100),
    hc=['opt'],
    q=np.logspace(0.1, 5, 100000),
    #q=100000,
)
```

```python
plt.plot(theta_0 + vs, qs)
```
