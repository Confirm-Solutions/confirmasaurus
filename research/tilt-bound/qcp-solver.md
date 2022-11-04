---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.10.5 ('confirm')
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import confirm.mini_imprint.bound.binomial as binomial
from pyimprint.grid import make_cartesian_grid_range
```

## Optimizing q for Exponential Holder Bound

```python
def A_cp(n, t):
    return n * cp.sum(cp.logistic(t))


def opt_q_cp(n, theta_0, v, a):
    """
    CVXPY implementation of finding optimal q
    """
    A0 = binomial.A(n, theta_0)
    q = cp.Variable(pos=True)
    objective_fn = ((A_cp(n, theta_0 + (q + 1) * v) - A0) - np.log(a)) / (q + 1)
    objective = cp.Minimize(objective_fn)
    problem = cp.Problem(objective)
    problem.solve(qcp=True)
    return q.value + 1
```

```python
theta_0 = jnp.array([-2.0, -1.0, 0.3])
n = 350
f0 = 0.025
v = 0.1 * jnp.array([0.4, 1, -1.32])
```

```python
solver = binomial.ForwardQCPSolver(n)
solve_jit = jax.jit(solver.solve)
```

```python
%%time
q_opt = solve_jit(theta_0, v, f0)
```

```python
%%time
q_opt_qcp_cvxpy = opt_q_cp(n, theta_0, v, f0)
```

```python
qs = jnp.linspace(1.0001, 10, 100)
phis = np.array([solver.objective(q, theta_0, v, f0) for q in qs])
plt.plot(qs, phis)
plt.plot(q_opt, solver.objective(q_opt, theta_0, v, f0), "bo")
plt.plot(q_opt_qcp_cvxpy, solver.objective(q_opt_qcp_cvxpy, theta_0, v, f0), "r^")
```

```python
solver = binomial.ForwardQCPSolver(n)
solve_vmap_jit = jax.jit(jax.vmap(solver.solve, in_axes=(0, 0, None)))


def vectorize_run(key, m, d, a=0.025, n=350):
    theta_0 = jax.random.normal(key, (m, d))
    _, key = jax.random.split(key)
    v = 0.001 * jax.random.normal(key, (m, d))
    return solve_vmap_jit(theta_0, v, a)
```

```python
%%time
qs = vectorize_run(
    jax.random.PRNGKey(10),
    10000,
    3,
)
```

```python
plt.hist(qs[~np.isinf(qs)], bins=30)
print(np.sum(np.isinf(qs)))
```

## Optimizing q for Implicit Exponential Holder Bound

```python
def qcp_solve_bwd(n, theta_0, v, alpha):
    A0 = binomial.A(n, theta_0)
    shift = (binomial.A(n, theta_0 + v) - A0) + np.log(alpha)
    q = cp.Variable(pos=True)
    objective_fn = ((A_cp(n, theta_0 + (q + 1) * v) - A0) - (q + 1) * shift) / q
    objective = cp.Minimize(objective_fn)
    problem = cp.Problem(objective)
    problem.solve(qcp=True)
    return q.value + 1, problem.value
```

```python
n = 350
theta_0 = -0.1
v = 0.005
alpha = 0.025
```

```python
solver_bwd = binomial.BackwardQCPSolver(n)
solve_bwd_jit = jax.jit(solver_bwd.solve)
```

```python
solve_bwd_vmap_jit = jax.jit(jax.vmap(solver_bwd.solve, in_axes=(0, 0, None)))


def vectorize_run_bwd(key, m, d, a=0.025, n=350):
    theta_0 = jax.random.normal(key, (m, d))
    _, key = jax.random.split(key)
    v = 0.001 * jax.random.normal(key, (m, d))
    return solve_bwd_vmap_jit(theta_0, v, a)
```

```python
%%time
qs = vectorize_run_bwd(jax.random.PRNGKey(69), 10000, 3)
```

```python
plt.hist(qs, bins=50)
plt.show()
```

```python
%%time
opt_q = solve_bwd_jit(theta_0, v, alpha)
```

```python
opt_bound = binomial.q_holder_bound_bwd(opt_q, n, theta_0, v, alpha)

# brute force search method
qs = np.linspace(1.01, 1000, 1000)
bound_bwd_f = jax.vmap(binomial.q_holder_bound_bwd, in_axes=(0, None, None, None, None))
bounds = bound_bwd_f(qs, n, theta_0, v, alpha)
i_max = np.argmax(bounds)

# plot
plt.plot(qs, bounds)
plt.plot(qs[i_max], bounds[i_max], "r^")
plt.plot(opt_q, opt_bound, "b.")
plt.show()

print(bounds[i_max], opt_bound)
```

## Combine Both Forward and Backward

```python
n = 350
theta_0 = -0.1
v = 0.01
alpha = 0.005
```

```python
# Backward solve the implicit bound at theta_0
solver_bwd = binomial.BackwardQCPSolver(n)
opt_q_bwd = solver_bwd.solve(theta_0, v, alpha)
alpha_prime = binomial.q_holder_bound_bwd(opt_q_bwd, n, theta_0, v, alpha)

# Forward evaluate to get bound on f(theta_0 + v)
bound = binomial.q_holder_bound_fwd(opt_q_bwd, n, theta_0, v, alpha_prime)
bound, alpha, alpha_prime
```

## Optimize for v

```python
n = 350
theta_0 = -1.0
vs = np.linspace(-1, 1, 100)
bound_vs_f = jax.vmap(binomial.q_holder_bound_fwd, in_axes=(None, None, None, 0, None))
bound_vs = bound_vs_f(2, n, theta_0, vs, f0)
```

```python
plt.plot(vs, bound_vs)
```

```python
q = 20


def g(v, theta_0, q):
    return jnp.sum(
        binomial.logistic(theta_0 + q * v) / q - binomial.logistic(theta_0 + v)
    )


theta_0 = jnp.array([-0.2, -0.1])
d = theta_0.shape[0]
v_1d_len = 100
vs = (
    make_cartesian_grid_range(v_1d_len, -10 * np.ones(d), 10 * np.ones(d), 0).thetas().T
)
g_vmap = jax.vmap(g, in_axes=(0, None, None))
gs = g_vmap(vs, theta_0, q)

# find max
i_max = jnp.argmax(gs)

sc = plt.scatter(vs[:, 0], vs[:, 1], c=gs)
plt.scatter(vs[i_max, 0], vs[i_max, 1], c="r")
plt.colorbar(sc)
plt.show()
```

## Tile-based

```python
tile_solver = binomial.TileForwardQCPSolver(
    n=350,
)
```

```python
theta_0 = jnp.array([-2.0, -1.0, 0.3])
n = 350
f0 = 0.025
radius = 0.05
v_coords = [[-1.0, 1.0]] * theta_0.shape[0]
mgrid = np.meshgrid(*v_coords, indexing="ij")
vs = radius * np.concatenate([coord.reshape(-1, 1) for coord in mgrid], axis=1)
```

```python
q_opt = tile_solver.solve(theta_0, vs, f0)
```

```python
qs = jnp.linspace(1, jnp.maximum(2, q_opt) + 10, 1000)
objs = jax.vmap(
    binomial.tilt_bound_fwd_tile,
    in_axes=(0, None, None, None, None),
)(qs, n, theta_0, vs, f0)
plt.plot(qs, objs)
obj_opt = binomial.tilt_bound_fwd_tile(q_opt, n, theta_0, vs, f0)
plt.scatter(q_opt, obj_opt, color="r")
print(jnp.min(objs), obj_opt)
print(qs[jnp.argmin(objs)], q_opt)
```

```python
f_opt = jax.jit(
    jax.vmap(
        tile_solver.solve,
        in_axes=(0, None, None),
    )
)
thetas = jnp.repeat(theta_0[None], 10000, axis=0)
```

```python
%%time
f_opt(thetas, vs, f0)
```

```python
tile_bwd_solver = binomial.TileBackwardQCPSolver(n=n)
```

```python
q_opt = tile_bwd_solver.solve(
    theta_0,
    vs,
    f0,
)
```

```python
qs = jnp.linspace(1, jnp.maximum(2, q_opt) + 10, 1000)
objs = jax.jit(
    jax.vmap(
        binomial.tilt_bound_bwd_tile,
        in_axes=(0, None, None, None, None),
    )
)(qs, n, theta_0, vs, f0)
plt.plot(qs, objs)
obj_opt = binomial.tilt_bound_bwd_tile(q_opt, n, theta_0, vs, f0)
plt.scatter(q_opt, obj_opt, color="r")
print(jnp.max(objs), obj_opt)
print(qs[jnp.argmax(objs)], q_opt)
```
