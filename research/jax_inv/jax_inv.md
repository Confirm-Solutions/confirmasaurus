---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('imprint')
    language: python
    name: python3
---

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax_explicit_inv import *
from jax.config import config

# This line is critical for enabling 64-bit floats.
config.update("jax_enable_x64", True)
```

```python
mats = np.random.rand(int(1e6), 4, 4)
```

```python
%%time
_ = np.linalg.inv(mats)
```

```python
%%time
_ = jnp.linalg.inv(mats)
```

```python
def bench(f, N, d, iters=5):
    np.random.seed(10)
    mats = np.random.rand(int(N), d, d)
    jax_mats = jnp.array(mats)
    for i in range(iters):
        start = time.time()
        correct = jnp.linalg.inv(jax_mats).block_until_ready()
        end = time.time()
    jli_time = end - start
    for i in range(iters):
        start = time.time()
        fout = f(jax_mats).block_until_ready()
        end = time.time()
    f_time = end - start
    # np.testing.assert_allclose(fout, correct, rtol=1e-4)
    return jli_time / N * 1e6, f_time / N * 1e6
```

```python
mats = np.random.rand(int(1e4), 15, 15)
```

```python
%%timeit -n 10
vmap_inv_recurse(mats)
```

```python
bench(vmap_inv44, 1e5, 4), bench(vmap_inv33, 3e5, 3), bench(vmap_inv22, 1e6, 2)
```

```python
mats.shape
```

```python
for d in range(1, 12):
    n = 6e5 / (d ** 2)
    print(d, n)
    print(bench(vmap_inv_recurse, n, d, iters=2))
```

## Code generation?!

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax_explicit_inv import *
from jax.config import config

# This line is critical for enabling 64-bit floats.
config.update("jax_enable_x64", True)
```

```python
orig = np.random.rand(3,3)
U = orig.copy()
L = np.diag(np.ones(U.shape[0]))
for k in range(U.shape[0] - 1):
    invkk = 1.0 / U[k,k]
    L[(k+1):,k] = U[(k + 1):,k].copy() * invkk
    U[(k+1):,k:] -= U[k:(k+1),k:] * U[(k+1):,k:(k+1)] * invkk
LU = U.copy()
LL = L.copy()
np.fill_diagonal(LL, 0)
soln = LU + LL
np.testing.assert_allclose(L.dot(U), orig)
```

```python
from dataclasses import dataclass
class CodeGenCtx:
    def __init__(self):
        self.assignments = []
        self.definitions = dict()

    def assign(self, name, definition):
        self.assignments.append(name)
        self.definitions[name] = definition
        return self.assignments[-1]
        
    def lines(self):
        return [f'{a} = {self.definitions[a]}' for a in self.assignments]
```

```python
def gen_lu(ctx, M, d):
    U = copy.deepcopy(M)
    L = [[None] * d for i in range(d)]
    for k in range(d - 1):
        inv_k = ctx.assign(f'inv_{k}', f'1.0 / {U[k][k]}')
        for j in range(k + 1, d):
            L[j][k] = ctx.assign(f'L_{j}{k}', f'{U[j][k]} * {inv_k}')
        for i in range(k + 1, d):
            for j in range(k + 1, d):
                if i == k + 1:
                    name = f'U_{i}{j}'
                else:
                    name = f'U_{k}_{i}{j}'
                U[i][j] = ctx.assign(name, f'{U[i][j]} - {U[k][j]} * {U[i][k]} * {inv_k}')
    LU = [[U[i][j] if i <= j else L[i][j] for j in range(d)] for i in range(d)]
    return LU
```

```python
def build_linalg(name, generator, d, print_code=True):
    ctx = CodeGenCtx()
    M = [[f'm[{i}, {j}]' for j in range(d)] for i in range(d)]
    LU = generator(ctx, M, d)
    lines = ctx.lines()
    lines.append('return jnp.array([' + ', '.join([
        '[' + ', '.join(LU[i]) + ']'
        for i in range(d)
    ]) + '])')
    lines = [f'def {name}(m):'] + ['    ' + l for l in lines] 
    code = '\n'.join(lines)
    if print_code:
        print(code)
    return code

exec(build_linalg('LU_decomp', gen_lu, 3))
np.testing.assert_allclose(LU_decomp(orig), soln)
```

```python
def gen_upper_tri_inv(ctx, U, d):
    invU = copy.deepcopy(U)
    for k in range(d)[::-1]:
        invU[k][k] = ctx.assign(f'invU_{k}{k}', f'1.0 / {invU[k][k]}')
        for j in range(k + 1, d):
            invU[k][j] = ctx.assign(f'invU_{k}{j}', f'{invU[k][j]} * {invU[k][k]}')
        for i in range(k):
            mult = f'-{invU[i][k]}'
            invU[i][k] = ctx.assign(f'invU_{k}_{i}{k}', f'{mult} * {invU[k][k]}')
            for j in range(k + 1, d):
                invU[i][j] = ctx.assign(f'invU_{k}_{i}{j}', f'{invU[i][j]} + {mult} * {invU[k][j]}')
    return invU
exec(build_linalg('upper_tri_inv', gen_upper_tri_inv, 3))
np.testing.assert_allclose(np.triu(upper_tri_inv(soln)), np.linalg.inv(np.triu(soln)))
```

```python
def transpose(A):
    d = len(A)
    return [[A[j][i] for j in range(d)] for i in range(d)]

def gen_lu_inv(ctx, LU, d):
    invU = copy.deepcopy(LU)
    for k in range(d)[::-1]:
        invU[k][k] = ctx.assign(f'invU_{k}{k}', f'1.0 / {invU[k][k]}')
        for j in range(k + 1, d):
            invU[k][j] = ctx.assign(f'invU_{k}{j}', f'{invU[k][j]} * {invU[k][k]}')
        for i in range(k):
            mult = f'-{invU[i][k]}'
            invU[i][k] = ctx.assign(f'invU_{k}_{i}{k}', f'{mult} * {invU[k][k]}')
            for j in range(k + 1, d):
                invU[i][j] = ctx.assign(f'invU_{k}_{i}{j}', f'{invU[i][j]} + {mult} * {invU[k][j]}')
                
    invLU_T = transpose(invU)
    for i in range(d - 1):
        for j in range(i + 1, d):
            invLU_T[i][j] = '0'
    for k in range(d)[::-1]:
        for i in range(k):
            mult = f'-{LU[k][i]}'
            for j in range(d):
                name = f'invLU_T_{k}_{i}{j}'
                invLU_T[i][j] = ctx.assign(name, f'{invLU_T[i][j]} + {mult} * {invLU_T[k][j]}')
    return transpose(invLU_T)

exec(build_linalg('lu_inv', gen_lu_inv, 3))
np.testing.assert_allclose(lu_inv(LU_decomp(orig)), np.linalg.inv(orig), rtol=1e-5)
```

```python
def gen_lu_solve(ctx, LU, B, d):
    Y = [None] * d
    for i in range(d):
        terms_i = [f'-{LU[i][j]}*{Y[j]}' for j in range(i)]
        Y[i] = ctx.assign(f'Y_{i}', f'{B[i]}' + ''.join(terms_i))
    X = [None] * d
    for i in range(d)[::-1]:
        invkk = ctx.assign(f'inv_{i}', f'1.0 / {LU[i][i]}')
        terms_i = [f'-{LU[i][j]}*{X[j]}*{invkk}' for j in range(i + 1, d)]
        X[i] = ctx.assign(f'X_{i}', f'{Y[i]}*{invkk}' + ''.join(terms_i))
    return X
```

```python
def gen_solve(ctx, M, Y, d):
    LU = gen_lu(ctx, M, d)
    return gen_lu_solve(ctx, LU, Y, d)
    
def build_linalg_solve(name, generator, d, print_code=True):
    ctx = CodeGenCtx()
    M = [[f'm[{i}, {j}]' for j in range(d)] for i in range(d)]
    Y = [f'y[{i}]' for i in range(d)]
    X = gen_solve(ctx, M, Y, d)
    lines = ctx.lines()
    lines.append('return jnp.array([' + ', '.join(X) + '])')
    lines = [f'def {name}(m, y):'] + ['    ' + l for l in lines] 
    code = '\n'.join(lines)
    if print_code:
        print(code)
    return code
exec(build_linalg_solve('solve3', gen_solve, 3))
```

```python
np.random.seed(0)
A = np.random.rand(3,3)
y = np.random.rand(3)
np.testing.assert_allclose(solve(A, y), np.linalg.solve(A,y))
```

```python
def gen_inv(ctx, M, d):
    LU = gen_lu(ctx, M, d)
    out = gen_lu_inv(ctx, LU, d)
    return out
```

```python
def bench_solve(f, N, d, iters=5):
    np.random.seed(10)
    mats = np.random.rand(int(N), d, d)
    bs = np.random.rand(int(N), d)
    jax_mats = jnp.array(mats)
    for i in range(iters):
        start = time.time()
        correct = jnp.linalg.solve(jax_mats, bs).block_until_ready()
        end = time.time()
    jli_time = end - start
    for i in range(iters):
        start = time.time()
        fout = f(jax_mats, bs).block_until_ready()
        end = time.time()
    f_time = end - start
    # np.testing.assert_allclose(fout, correct, rtol=1e-4)
    return jli_time / N * 1e6, f_time / N * 1e6
```

```python
for d in range(1, 12):
    exec(build_linalg(f'inv{d}', gen_inv, d, print_code=False))
    exec(build_linalg(f'lu{d}', gen_lu, d, print_code=False))
    exec(build_linalg_solve(f'solve{d}', gen_solve, d, print_code=False))
    f = globals()[f'inv{d}']
    f_lu = globals()[f'lu{d}']
    f_solve = globals()[f'solve{d}']
    mat = np.random.rand(d, d)
    b = np.random.rand(d)
    np.testing.assert_allclose(f(mat), np.linalg.inv(mat), rtol=1e-5)
    np.testing.assert_allclose(f_solve(mat, b), np.linalg.solve(mat, b), rtol=1e-5)
    vmap = jax.jit(jax.vmap(f))
    vmap_lu = jax.jit(jax.vmap(f_lu))
    vmap_solve = jax.jit(jax.vmap(f_solve))
    globals()[f'vmap_inv{d}'] = vmap
    print('\n', d)
    n = 1e5 / (d ** 2)
    print('recursive + cramer', bench(vmap_inv_recurse, n, d))
    print('code gen', bench(vmap, n, d))
    print('lu gen', bench(vmap_lu, n, d))
    print('solve', bench_solve(vmap_solve, n, d))
```

```python

```

```python
def inv_jax_lax(m):
    lu, pivot, perm = jax.lax.linalg.lu(m)
    U_inv = jax.lax.linalg.triangular_solve(lu, jnp.diag(np.array([1.0,1,1])), lower=False, unit_diagonal=False)
    full_inv = jax.lax.linalg.triangular_solve(lu, U_inv, lower=True, unit_diagonal=True)
    return full_inv[:, perm]
vmap_inv_jax_lax = jax.jit(jax.vmap(inv_jax_lax))
bench(vmap_inv_jax_lax, 1e4, 3)
```

```python
%%time
_ = vmap_inv_jax(mats)
```

## JAX is fast for large matrices

```python
A = np.random.rand(5000, 5000)
```

```python
flops = 5000 ** 3 * 2 / 3.
flops * 0.3 / 1e9 / 10 / 2
```

```python
%%time
np.linalg.inv(A)
```

```python
%%time
jnp.linalg.inv(jnp.array(A, dtype=jnp.float64))
```

```python

```
