from functools import partial

import jax
import jax.numpy as jnp


# This code is copied from scipy.interpolate.interpn and modified for working with JAX.
def _find_indices(grid, xi):

    # find relevant edges between which xi are situated
    indices = []
    # compute distance to lower edge in unity units
    norm_distances = []

    for i in range(len(grid)):
        g = grid[i]
        idx = jnp.searchsorted(g, xi[i]) - 1
        idx = jnp.where(idx > 0, idx, 0)
        idx = jnp.where(idx > g.size - 2, g.size - 2, idx)
        indices = indices + [idx]
        denom = g[idx + 1] - g[idx]
        norm_distances = norm_distances + [
            jnp.where(denom != 0, (xi[i] - g[idx]) / denom, 0)
        ]

    indices = jnp.array(indices)
    norm_distances = jnp.array(norm_distances)
    return indices, norm_distances


def _evaluate_linear(grid, values, indices, norm_distances):
    lr_grid = jnp.meshgrid(
        *[jnp.array([0, 1]) for i in range(len(grid))], indexing="ij"
    )

    hypercube_indices = [
        jnp.array([indices[i], indices[i] + 1])[lr_grid[i]] for i in range(len(grid))
    ]
    hypercube_weights = [
        jnp.array([1 - norm_distances[i], norm_distances[i]])[lr_grid[i]]
        for i in range(len(grid))
    ]

    hypercube_indices_ravel = jnp.stack(hypercube_indices, axis=-1).reshape(
        (-1, len(grid))
    )
    hypercube_weights_ravel = jnp.stack(hypercube_weights, axis=-1).reshape(
        (-1, len(grid))
    )
    hypercube_flat_idx = jnp.ravel_multi_index(
        hypercube_indices_ravel.T,
        [grid[i].shape[0] for i in range(len(grid))],
        mode="clip",
    )
    values_ravel = values.ravel()
    return (
        values_ravel[hypercube_flat_idx] * hypercube_weights_ravel.prod(axis=-1)
    ).sum()


@jax.jit
@partial(jax.vmap, in_axes=(None, None, 0))
def interpn(points, values, xi):
    grid = tuple([jnp.asarray(p) for p in points])
    indices, norm_distances = _find_indices(grid, xi)
    return _evaluate_linear(grid, values, indices, norm_distances)
