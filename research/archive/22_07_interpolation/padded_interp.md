```python
from outlaw.nb_util import setup_nb

setup_nb()
```

```python
from functools import partial, reduce
import outlaw.inla as inla
import outlaw.quad as quad
import outlaw.berry as berry
import outlaw.interp as interp
import numpy as np
from numpy import nan
import jax
import jax.numpy as jnp
```

## Version 1: different size tables

```python
from dataclasses import dataclass


def f(n1, n2, pts):
    x, y = pts
    x /= n1
    y /= n2
    return x**2 + y**2


fv = jax.vmap(f, in_axes=(None, None, 0))

nr = jnp.arange(10, 41, 5)
max_nr = int(jnp.max(nr))


def hash_n(n1, n2):
    return n1 * max_nr + n2


def make_table(n1, n2):
    grids_1d = [jnp.arange(0, nv + 1, 2) for nv in [n1, n2]]
    pts = jnp.stack(jnp.meshgrid(*grids_1d, indexing="ij"), axis=-1)
    pts_2d = pts.reshape((-1, 2))
    values = fv(n1, n2, pts_2d).reshape(pts.shape[:-1])
    return (hash_n(n1, n2), grids_1d, values)


tables = dict()
for n1 in nr:
    for n2 in nr:
        T = make_table(n1, n2)
        tables[int(T[0])] = T
```

```python
@jax.jit
def interp_table(n1, n2, xi):
    def interp_wrapper(T):
        return jnp.where(T[0] == hash_n(n1, n2), interp.interpn(*T[1:], xi), 0)

    def interp_one_table(T):
        # check whether we should ignore this table or not.
        cond = T[0] == hash_n(n1, n2)
        return jax.lax.cond(cond, lambda: interp_wrapper(T), lambda: 0.0)

    # check each table.
    interps = jax.tree_util.tree_map(
        interp_one_table, tables, is_leaf=lambda x: not isinstance(x, dict)
    )
    # sum the results. the interpolation result will be zero for every table except the one
    return jnp.sum(jnp.array(jax.tree_util.tree_leaves(interps)))


xi = jnp.array([0.5, 0.5])
interp_table(10, 15, xi)
```

```python
def gen_pts(N):
    keys = jax.random.split(jax.random.PRNGKey(0), N * 2)
    uniforms = jax.vmap(jax.random.uniform)(keys)
    rand_pts = (uniforms.reshape((-1, 2)) * 20).astype(int)
    return rand_pts


N = int(1e6)
rand_pts = gen_pts(N)
interp_ex = interp.interpnv(*tables[hash_n(10, 15)][1:], rand_pts)
```

```python
%%time
rand_pts = gen_pts(N)
```

```python
%%time
_ = interp.interpnv(*tables[hash_n(10, 15)][1:], rand_pts)
```

```python
def interpn_randoms(k1, k2, k3, k4, lookup_n):
    n1 = jax.random.choice(k3, nr)
    n2 = jax.random.choice(k4, nr)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)
    x = (u1 * n1).astype(int)
    y = (u2 * n2).astype(int)
    if lookup_n:
        return interp_table(n1, n2, jnp.array([x, y]))
    else:
        return interp.interpn(*tables[hash_n(40, 40)][1:], jnp.array([x, y]))


@partial(jax.jit, static_argnums=(0,))
def interpn_all(lookup_n):
    k = jax.random.PRNGKey(0)
    keys = jax.random.split(k, 3 * N).reshape((-1, 3, 2))
    return jax.vmap(interpn_randoms, in_axes=(0, 0, 0, 0, None))(
        keys[:, 0, :], keys[:, 1, :], keys[:, 2, :], keys[:, 3, :], lookup_n
    )


interpn_all(False)
interpn_all(True)
```

```python
%%time
_ = interpn_all(False)
```

```python
%%time
_ = interpn_all(True)
```

## Version 2: Padding tables

```python
from dataclasses import dataclass


def f(n1, n2, pts):
    x, y = pts
    x /= n1
    y /= n2
    return jax.lax.cond((x <= 1) & (y <= 1), lambda: x**2 + y**2, lambda: 0.0)


fv = jax.vmap(f, in_axes=(None, None, 0))

nr = jnp.arange(10, 41, 5).astype(float)
max_nr = int(jnp.max(nr))


def hash_n(n1, n2):
    return n1 * max_nr + n2


def make_table(n1, n2):
    # Note max_nr here instead of n1, n2
    grids_1d = [jnp.arange(0, max_nr + 1, 2) for _ in [n1, n2]]
    pts = jnp.stack(jnp.meshgrid(*grids_1d, indexing="ij"), axis=-1)
    pts_2d = pts.reshape((-1, 2))
    values = fv(n1, n2, pts_2d).reshape(pts.shape[:-1])
    return (hash_n(n1, n2), grids_1d, values)


tables = dict()
for n1 in nr:
    for n2 in nr:
        T = make_table(n1, n2)
        tables[int(T[0])] = T
```

```python
@jax.jit
def interp_table(n1, n2, xi):
    def interp_one_table(T):
        # check whether we should ignore this table or not.
        cond = T[0] == hash_n(n1, n2)
        return jax.lax.cond(cond, lambda: interp.interpn(*T[1:], xi), lambda: 0.0)

    table_defs = jax.tree_util.tree_leaves(
        tables, is_leaf=lambda x: not isinstance(x, dict)
    )

    def reduce_f(x, T):
        return x + interp_one_table(T)

    return reduce(reduce_f, table_defs, 0)
    # # check each table.
    # interps = jax.tree_util.tree_map(
    #     interp_one_table, tables, is_leaf=lambda x: not isinstance(x, dict)
    # )
    # # sum the results. the interpolation result will be zero for every table except the one
    # return jnp.sum(jnp.array(jax.tree_util.tree_leaves(interps)))


xi = jnp.array([0.5, 0.5])
interp_table(10, 15, xi)
```

```python
@jax.jit
def interp_table(n1, n2, xi):
    def interp_one_table(T):
        # check whether we should ignore this table or not.
        cond = T[0] == hash_n(n1, n2)
        return jax.lax.cond(cond, lambda: interp.interpn(*T[1:], xi), lambda: 0.0)

    table_defs = jax.tree_util.tree_leaves(
        tables, is_leaf=lambda x: not isinstance(x, dict)
    )

    def reduce_f(x, T):
        return x + interp_one_table(T)

    return reduce(reduce_f, table_defs, 0)
    # # check each table.
    # interps = jax.tree_util.tree_map(
    #     interp_one_table, tables, is_leaf=lambda x: not isinstance(x, dict)
    # )
    # # sum the results. the interpolation result will be zero for every table except the one
    # return jnp.sum(jnp.array(jax.tree_util.tree_leaves(interps)))


xi = jnp.array([0.5, 0.5])
interp_table(10, 15, xi)
```

```python
table_list = jax.tree_util.tree_leaves(
    tables, is_leaf=lambda x: not isinstance(x, dict)
)
```

```python
grids = table_list[0][1]
values_shape = table_list[0][2].shape
compact_tables = jnp.stack(
    [jnp.concatenate((jnp.array([T[0]]), T[2].flatten())) for T in table_list], axis=0
)
```

```python
@jax.jit
def interp_table(n1, n2, xi):
    idx = jnp.searchsorted(compact_tables[:, 0], hash_n(n1, n2))
    return interp.interpn(grids, compact_tables[idx, 1:].reshape(values_shape), xi)


xi = jnp.array([0.5, 0.5])
interp_table(10, 15, xi)
```

```python
def interpn_randoms(k1, k2, k3, k4, lookup_n):
    n1 = jax.random.choice(k3, nr)
    n2 = jax.random.choice(k4, nr)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)
    x = (u1 * n1).astype(int)
    y = (u2 * n2).astype(int)
    if lookup_n:
        return interp_table(n1, n2, jnp.array([x, y]))
    else:
        return interp.interpn(*tables[hash_n(40, 40)][1:], jnp.array([x, y]))


@partial(jax.jit, static_argnums=(0,))
def interpn_all(lookup_n):
    k = jax.random.PRNGKey(0)
    keys = jax.random.split(k, 3 * N).reshape((-1, 3, 2))
    return jax.vmap(interpn_randoms, in_axes=(0, 0, 0, 0, None))(
        keys[:, 0, :], keys[:, 1, :], keys[:, 2, :], keys[:, 3, :], lookup_n
    )


interpn_all(False)
interpn_all(True)
```

```python
%%time
_ = interpn_all(False)
```

```python
%%time
_ = interpn_all(True)
```

## Version 3: lax.scan?

```python
from dataclasses import dataclass


def f(n1, n2, pts):
    x, y = pts
    x /= n1
    y /= n2
    return jax.lax.cond((x <= 1) & (y <= 1), lambda: x**2 + y**2, lambda: 0.0)


fv = jax.vmap(f, in_axes=(None, None, 0))

nr = jnp.arange(10, 41, 5).astype(float)
max_nr = int(jnp.max(nr))


def hash_n(n1, n2):
    return n1 * max_nr + n2


def make_table(n1, n2):
    # Note max_nr here instead of n1, n2
    grids_1d = [jnp.arange(0, nv + 1, 2) for nv in [n1, n2]]
    pts = jnp.stack(jnp.meshgrid(*grids_1d, indexing="ij"), axis=-1)
    pts_2d = pts.reshape((-1, 2))
    values = fv(n1, n2, pts_2d).reshape(pts.shape[:-1])
    return (jnp.array([hash_n(n1, n2)]), grids_1d, values)


tables = []
for n1 in nr:
    for n2 in nr:
        T = make_table(n1, n2)
        tables.append(T)
```

```python
# @jax.jit
def interp_table(n1, n2, xi):
    def interp_one_table(T):
        # check whether we should ignore this table or not.
        cond = T[0] == hash_n(n1, n2)
        return jax.lax.cond(cond, lambda: interp.interpn(*T[1:], xi), lambda: 0.0)

    def scan_f(result, T):
        return (interp_one_table(T) + result, None)

    tables_flat = jax.tree_util.tree_flatten(
        tables, is_leaf=lambda x: not isinstance(x, list)
    )
    print(len(tables_flat))
    return jax.lax.scan(scan_f, 0, tables_flat)[0]
    # # check each table.
    # interps = jax.tree_util.tree_map(
    #     interp_one_table, tables, is_leaf=lambda x: not isinstance(x, dict)
    # )
    # # sum the results. the interpolation result will be zero for every table except the one
    # return jnp.sum(jnp.array(jax.tree_util.tree_leaves(interps)))


xi = jnp.array([0.5, 0.5])
interp_table(10, 15, xi)
```

```python
def interpn_randoms(k1, k2, k3, k4, lookup_n):
    n1 = jax.random.choice(k3, nr)
    n2 = jax.random.choice(k4, nr)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)
    x = (u1 * n1).astype(int)
    y = (u2 * n2).astype(int)
    if lookup_n:
        return interp_table(n1, n2, jnp.array([x, y]))
    else:
        return interp.interpn(*tables[hash_n(40, 40)][1:], jnp.array([x, y]))


@partial(jax.jit, static_argnums=(0,))
def interpn_all(lookup_n):
    k = jax.random.PRNGKey(0)
    keys = jax.random.split(k, 3 * N).reshape((-1, 3, 2))
    return jax.vmap(interpn_randoms, in_axes=(0, 0, 0, 0, None))(
        keys[:, 0, :], keys[:, 1, :], keys[:, 2, :], keys[:, 3, :], lookup_n
    )


interpn_all(False)
interpn_all(True)
```

```python
%%time
_ = interpn_all(False)
```

```python
%%time
_ = interpn_all(True)
```

## JUNK

```python
@jax.jit
def interp_table(n1, n2, xi):
    def interp_one_table(T):
        # check whether we should ignore this table or not.
        cond = T[0] == hash_n(n1, n2)
        return jax.lax.cond(
            cond,
            lambda: interp.interpn(grids, T[1:].reshape(values_shape), xi),
            lambda: 0.0,
        )

    def scan_f(result, T):
        return (interp_one_table(T) + result, None)

    return jax.lax.scan(scan_f, 0, compact_tables)[0]

    # # check each table.
    # interps = jax.tree_util.tree_map(
    #     interp_one_table, tables, is_leaf=lambda x: not isinstance(x, dict)
    # )
    # # sum the results. the interpolation result will be zero for every table except the one
    # return jnp.sum(jnp.array(jax.tree_util.tree_leaves(interps)))
```

```python
# tables_keys = list(tables.keys())

# # @jax.jit
# def interp_table(n1, n2, xi):
#     def interp_wrapper(T):
#         return jnp.where(T[0] == hash_n(n1, n2), interp.interpn(*T[1:], xi), 0)

#     def get_interp_data(i, T):
#         # check whether we should ignore this table or not.
#         cond = T[0] == hash_n(n1, n2)
#         return jax.lax.cond(
#             cond,
#             lambda: T[i],
#             lambda: jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), T[i])
#         )

#     def cond(i):
#         return tables[tables_keys[i]] == hash_n(n1, n2)

#     def body(i):
#         return i + 1

#     init_val = 0

#     idx = jax.lax.while_loop(cond, body, init_val)

#     # TODO: there's probably a way to do this with just one tree_map.
#     grids = jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(
#         partial(get_interp_data, 0), tables, is_leaf=lambda x: not isinstance(x, dict)
#     )))
#     values = jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(
#         partial(get_interp_data, 1), tables, is_leaf=lambda x: not isinstance(x, dict)
#     )))
#     print(grids.shape, values.shape)
#     return interp.interpn(grids, values, xi)


# xi = jnp.array([0.5, 0.5])
# interp_table(10, 15, xi)
```
