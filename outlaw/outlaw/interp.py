from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
@partial(jax.vmap, in_axes=(None, None, 0))
def interpn(points, values, xi):
    """
    A JAX reimplementation of scipy.interpolate.interpn. Most of the input
    validity checks have been removed, so make sure your inputs are correct or
    go implement those checks yourself.

    In addition, the keyword arguments are:
    - `method="linear"`
    - `bounds_error=False`
    - `fill_value=None`

    The scipy source is here:
    https://github.com/scipy/scipy/blob/651a9b717deb68adde9416072c1e1d5aa14a58a1/scipy/interpolate/_rgi.py#L445-L614

    The original docstring from scipy:
    Multidimensional interpolation on regular or rectilinear grids.

    Strictly speaking, not all regular grids are supported - this function
    works on *rectilinear* grids, that is, a rectangular grid with even or
    uneven spacing.

    Args:
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions. The points in
            each dimension (i.e. every elements of the points tuple) must be
            strictly ascending or descending.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions. Complex data can be
            acceptable.
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

    Returns:
        values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
            Interpolated values at input coordinates.
    """

    grid = tuple([jnp.asarray(p) for p in points])
    indices, norm_distances = _find_indices(grid, xi)
    return _evaluate_linear(grid, values, indices, norm_distances)


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
    d = len(grid)
    # Construct the unit d-dimensional cube.
    unit_cube = jnp.meshgrid(*[jnp.array([0, 1]) for i in range(d)], indexing="ij")

    # Choose the left or right index for each corner of the hypercube. these
    # are 1D indices which get used in will later be used to construct the ND
    # indices of each corner.
    hypercube_dim_indices = [
        jnp.array([indices[i], indices[i] + 1])[unit_cube[i]] for i in range(d)
    ]
    # the final indices will be the unraveled ND indices produced from the 1D
    # indices above.
    hypercube_indices = jnp.ravel_multi_index(
        [hypercube_dim_indices[i].flatten() for i in range(d)],
        [grid[i].shape[0] for i in range(d)],
        mode="clip",
    )

    # the weights for the left and right sides of each 1D interval.
    # norm_distance is the normalized distance from the left edge so the weight
    # will be (1 - norm_distance) for the left edge
    hypercube_dim_weights = [
        jnp.array([1 - norm_distances[i], norm_distances[i]])[unit_cube[i]]
        for i in range(d)
    ]
    # the final weights will be the product of the weights for each dimension
    hypercube_weights = jnp.prod(jnp.array(hypercube_dim_weights), axis=0).ravel()

    # finally, select the values to interpolate and multiply by the weights.
    return (values.ravel()[hypercube_indices] * hypercube_weights).sum()
