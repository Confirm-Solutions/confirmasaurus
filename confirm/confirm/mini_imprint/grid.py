from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import List

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class HyperPlane:
    """
    A plane defined by:
    x \cdot n - c = 0

    Sign convention: When used as the boundary between null hypothesis and
    alternative, the normal should point towards the null hypothesis space.
    """

    n: np.ndarray
    c: float


@dataclass
class Grid:
    """
    The first two arrays define the grid points/tiles:
    - thetas: the center of each hyperrectangle.
    - radii: the half-width of each hyperrectangle in each dimension.

    The next four arrays define the tiles:
    - grid_pt_idx is an array with an entry for each tile that contains to
      index of the original grid point from which that tile was created
    - null_truth indicates the truth of each null hypothesis for each tile.
    - null_hypos contains the hyperplanes that define the null hypotheses.
    """

    thetas: np.ndarray
    radii: np.ndarray
    null_truth: np.ndarray
    grid_pt_idx: np.ndarray
    null_hypos: List[HyperPlane] = field(default_factory=lambda: [])

    @property
    def n_tiles(self):
        return self.null_truth.shape[0]

    @property
    def vertices(self):
        center = self.thetas[self.grid_pt_idx]
        radii = self.radii[self.grid_pt_idx]
        return (
            center[:, None, :]
            + hypercube_vertices(self.d)[None, :, :] * radii[:, None, :]
        )

    @property
    def n_grid_pts(self):
        return self.thetas.shape[0]

    @property
    def theta_tiles(self):
        return self.thetas[self.grid_pt_idx]

    @property
    def d(self):
        return self.thetas.shape[-1]


def index_grid(g: Grid, idxs: np.ndarray):
    """
    Take a subset of a grid by indexing into the tiles.

    Note: the grid points are not modified, so the resulting grid may have
    unused grid point.

    Args:
        g: the grid
        idxs: the tiles indexer

    Returns:
        the Grid subset.
    """
    return Grid(
        g.thetas,
        g.radii,
        g.null_truth[idxs],
        g.grid_pt_idx[idxs],
        g.null_hypos,
    )


def concat_grids(*gs: List[Grid], shared_theta=False):
    """
    Concat a list of grids.

    Note: this assumes the grids have the same null hypotheses. Concatenating
    grids with different null hypotheses doesn't make sense and is not
    supported.

    Args:
        shared_theta: Do the grids already share the same grid points. This can
            be useful if you are combining `concat_grid` with `index_grid`.
            Defaults to False.

    Returns:
        The concatenated grid.
    """
    if len(gs) == 1:
        return gs[0]

    null_truth = np.concatenate([g.null_truth for g in gs], axis=0)

    if shared_theta:
        thetas = gs[0].thetas
        radii = gs[0].radii
        grid_pt_idx = np.concatenate([g.grid_pt_idx for g in gs], axis=0)
    else:
        thetas = np.concatenate([g.thetas for g in gs], axis=0)
        radii = np.concatenate([g.radii for g in gs], axis=0)
        grid_pt_offset = np.concatenate(([0], np.cumsum([g.n_grid_pts for g in gs])))
        grid_pt_idx = np.concatenate(
            [g.grid_pt_idx + grid_pt_offset[i] for i, g in enumerate(gs)], axis=0
        )
    return Grid(
        thetas,
        radii,
        null_truth,
        grid_pt_idx,
        null_hypos=gs[0].null_hypos,
    )


def plot_grid2d(g: Grid, null_hypos: List[HyperPlane] = [], dims=(0, 1)):
    """
    Plot a 2D grid.

    Args:
        g: the grid
        null_hypos: If provided, the function will plot red lines for the null
            hypothesis boundaries. Defaults to [].
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    vertices = g.vertices[..., dims]

    polys = []
    vertices = g.vertices()
    for i in range(g.n_tiles):
        vs = vertices[i]
        vs = vs[~np.isnan(vs).any(axis=1)]
        centroid = np.mean(vs, axis=0)
        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        order = np.argsort(angles)
        polys.append(mpl.patches.Polygon(vs[order], fill=None, edgecolor="k"))
        plt.text(*centroid, str(i))

    plt.gca().add_collection(
        mpl.collections.PatchCollection(polys, match_original=True)
    )

    maxvs = np.max(g.thetas, axis=0) + np.max(g.radii, axis=0)
    minvs = np.min(g.thetas, axis=0) - np.max(g.radii, axis=0)
    view_center = 0.5 * (maxvs + minvs)
    view_radius = (maxvs - minvs) * 0.55
    xlims = view_center[0] + np.array([-1, 1]) * view_radius[0]
    ylims = view_center[1] + np.array([-1, 1]) * view_radius[1]
    plt.xlim(xlims)
    plt.ylim(ylims)

    for h in null_hypos:
        if h.n[0] == 0:
            xs = np.linspace(*xlims, 100)
            ys = (h.c - xs * h.n[0]) / h.n[1]
        else:
            ys = np.linspace(*ylims, 100)
            xs = (h.c - ys * h.n[1]) / h.n[0]
        plt.plot(xs, ys, "r-")


def intersect_grid(g_in: Grid, null_hypos: List[HyperPlane], jit=False):
    """
    Intersect a grid with a set of null hypotheses. Tiles that cross the null
    hypothesis boundary are copied.

    Args:
        g_in: The input grid.
        null_hypos: The null hypotheses to intersect with. Sign convention:
            When used as the boundary between null hypothesis and alternative,
            the normal of a HyperPlane should point towards the null hypothesis
            space.
        jit: Should we jax.jit helper functions? This can make performance
            slower for small grids and much faster for large grids. Defaults to
            False.

    Returns:
        The intersected grid with copy tiles.
    """
    if len(null_hypos) == 0:
        return g_in

    eps = 1e-15
    n_grid_pts, n_params = g_in.thetas.shape

    ########################################
    # Step 1. Check for grid point intersection:
    # This is a rough check because it assumes all the tiles are
    # hyperrectangles.
    ########################################
    Hns = np.array([H.n for H in null_hypos])
    Hcs = np.array([H.c for H in null_hypos])

    gridpt_dist = g_in.thetas.dot(Hns.T) - Hcs[None]
    sphere_radii = np.sqrt(np.sum(g_in.radii**2, axis=-1))
    gridpt_any_intersect = np.any(np.abs(gridpt_dist) < sphere_radii[:, None], axis=-1)
    any_intersect = gridpt_any_intersect[g_in.grid_pt_idx]
    no_intersect = ~any_intersect
    tile_rough_dist = gridpt_dist[g_in.grid_pt_idx]
    null_truth = no_intersect[..., None] & (tile_rough_dist >= 0)

    ########################################
    # Step 2. Do a more precise check for tile intersections.
    # We also record whether the null hypothesis is true for each tile.
    ########################################

    def _precise_check_for_intersections(vertices, Hns, Hcs):
        n_tiles = vertices.shape[0]
        n_hypos = len(Hns)
        null_truth = jnp.full((n_tiles, n_hypos), -1)
        all_dist = vertices.dot(Hns.T) - Hcs[None, None]
        null_truth = ((all_dist >= 0) | jnp.isnan(all_dist)).all(axis=1)

        # 0 means alt true, 1 means null true
        all_above = ((all_dist >= -eps) | jnp.isnan(all_dist)).all(axis=1)
        all_below = ((all_dist <= eps) | jnp.isnan(all_dist)).all(axis=1)
        any_intersections = jnp.any(~(all_above | all_below), axis=1)

        return any_intersections, null_truth

    if jit:
        _precise_check_for_intersections = jax.jit(_precise_check_for_intersections)

    unit_vs = hypercube_vertices(g_in.d)
    intersect_gridpt = g_in.grid_pt_idx[any_intersect]
    intersect_vertices = g_in.thetas[intersect_gridpt, None, :] + (
        unit_vs[None, :, :] * g_in.radii[intersect_gridpt, None, :]
    )

    precise_any_intersect, precise_null_truth = _precise_check_for_intersections(
        intersect_vertices, Hns, Hcs
    )
    null_truth[any_intersect] = precise_null_truth
    any_intersect[any_intersect] = precise_any_intersect

    # the subset of the grid that does not need to be checked for intersection.
    g_ignore = index_grid(g_in, ~any_intersect)

    # the working subset that we *do* need to check for intersection.
    g = index_grid(g_in, any_intersect)

    full_null_truth = np.concatenate((g_in.null_truth, null_truth), axis=1)
    g_ignore.null_truth = full_null_truth[~any_intersect]
    g.null_truth = full_null_truth[any_intersect]

    if g.n_tiles == 0:
        return g_ignore

    for iH, H in enumerate(null_hypos):
        if iH == 0:
            vertices = intersect_vertices
        else:
            vertices = g.thetas[g.grid_pt_idx, None, :] + (
                unit_vs[None, :, :] * g.radii[g.grid_pt_idx, None, :]
            )

        ########################################
        # Step 3. Find any intersections for this null hypothesis.
        ########################################

        # Measure the distance of each vertex from the null hypo boundary
        # it's important to allow nan dist because some tiles may not have
        # every vertex slot filled. Unused vertex slots will contain nans.
        dist = vertices.dot(H.n) - H.c

        is_null = ((dist >= 0) | np.isnan(dist)).all(axis=1)

        # 0 means alt true, 1 means null true
        g.null_truth[is_null, iH] = 1
        g.null_truth[~is_null, iH] = 0

        # Identify the tiles to be copied by checking if all the tile vertices
        # are on the same side of the plane. Give some floating point slack
        # around zero so we don't suffer from imprecision.
        to_copy = ~(
            ((dist >= -eps) | np.isnan(dist)).all(axis=1)
            | ((dist <= eps) | np.isnan(dist)).all(axis=1)
        )
        copy_idxs = np.where(to_copy)[0]

        # The subset of the grid that we won't copy
        g_keep = index_grid(g, ~to_copy)

        ########################################
        # Step 4. Copy tiles.
        ########################################
        if copy_idxs.shape[0] == 0:
            g_copy = index_grid(g, np.s_[0:0])
        else:
            copy_null_truth = np.repeat(g.null_truth[copy_idxs], 2, axis=0)
            # If a tile is being copied that is because it intersects the null
            # hypo plane. So, one side should be null true and the other alt
            # true.
            copy_null_truth[::2, iH] = 1
            copy_null_truth[1::2, iH] = 0
            g_copy = Grid(
                g.thetas,
                g.radii,
                copy_null_truth,
                np.repeat(g.grid_pt_idx[copy_idxs], 2, axis=0),
                g.null_hypos,
            )

        # Hurray, we made it! We can concatenate our grids!
        g = concat_grids(g_keep, g_copy, shared_theta=True)

    # After all the copying is done, we can concat back to the tiles that we
    # ignored because we knew they would never be copied.
    out = concat_grids(g_ignore, g, shared_theta=True)
    out.null_hypos = out.null_hypos + null_hypos
    return out


def build_grid(
    thetas: np.ndarray,
    radii: np.ndarray,
    null_hypos: List[HyperPlane] = [],
    symmetry_planes: List[HyperPlane] = [],
    should_prune: bool = True,
):
    """
    Construct an Imprint grid from a set of grid point centers, radii and null
    hypothesis.
    1. Initially, we construct simple hyperrectangle cells.
    2. Then, we remove tiles on the negative side of any symmetry planes.
    3. Then, we copy cells that are intersected by the null hypothesis boundaries.
    4. Finally, we optionally remove tiles that are in the alternative
       hypothesis region for all null hypotheses. These tiles are not
       interesting for Type I Error analysis.

    Args:
        thetas: The centers of the hyperrectangle grid.
        radii: The half-width of each hyperrectangle in each dimension.
        null_hypos: A list of hyperplanes defining the boundary of the null
            hypothesis. The normal vector of these hyperplanes point into the null
            domain.
        symmetry_planes: A list of hyperplanes defining symmetry planes. These
            are used to filter out redundant tiles.
        should_prune: If True, remove tiles that are entirely in the alternative
            hypothesis space.


    Returns:
        a Grid object
    """
    n_grid_pts, _ = thetas.shape

    # Keep track of the various tile properties. See the Grid class docstring
    # for definitions.
    grid_pt_idx = np.arange(n_grid_pts)
    null_truth = np.full((n_grid_pts, 0), -1)

    g = Grid(thetas, radii, tile_vs, is_regular, null_truth, grid_pt_idx)
    g_sym = prune(intersect_grid(g, symmetry_planes), hard=True)
    g_sym.null_truth = np.empty((g_sym.n_tiles, 0), dtype=bool)
    g_sym.null_hypos = []
    g_out = intersect_grid(g_sym, null_hypos)
    if should_prune:
        return prune(g_out)
    else:
        return g_out

    # g = Grid(thetas, radii, null_truth, grid_pt_idx)
    # if len(null_hypos) > 0:
    #     g = intersect_grid(g, null_hypos)
    # return g


def cartesian_gridpts(theta_min, theta_max, n_theta_1d):
    """
    Produce a grid of points in the hyperrectangle defined by theta_min and
    theta_max.

    Args:
        theta_min: The minimum value of theta for each dimension.
        theta_max: The maximum value of theta for each dimension.
        n_theta_1d: The number of theta values to use in each dimension.

    Returns:
        theta: A 2D array of shape (n_grid_pts, n_params) containing the grid points.
        radii: A 2D array of shape (n_grid_pts, n_params) containing the
            half-width of each grid point in each dimension.
    """
    theta_min = np.asarray(theta_min)
    theta_max = np.asarray(theta_max)
    n_theta_1d = np.asarray(n_theta_1d)

    n_arms = theta_min.shape[0]
    theta1d = [
        np.linspace(theta_min[i], theta_max[i], 2 * n_theta_1d[i] + 1)[1::2]
        for i in range(n_arms)
    ]
    theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
    radii = np.empty(theta.shape)
    for i in range(theta.shape[1]):
        radii[:, i] = 0.5 * (theta1d[i][1] - theta1d[i][0])
    return theta, radii


def prune(g: Grid, hard: bool = False) -> Grid:
    """
    Remove tiles that are entirely within the alternative hypothesis space.

    Args:
        g: the Grid object
        hard: If True, remove tiles with any hypotheses in the alternative
            space. If False, remove tiles with all hypotheses in the alternative
            space.

    Returns:
        the pruned Grid object.
    """
    if g.null_truth.shape[1] == 0:
        return g
    if hard:
        keep = ~((g.null_truth == 0).any(axis=1))
    else:
        keep = ~((g.null_truth == 0).all(axis=1))
    return trim(index_grid(g, keep))


def trim(g: Grid) -> Grid:
    """
    Remove unused grid points from the grid.

    Args:
        g: The Grid to be trimmed.

    Returns:
        The trimmed Grid.
    """
    included_grid_pts, grid_pt_inverse = np.unique(g.grid_pt_idx, return_inverse=True)
    return Grid(
        g.thetas[included_grid_pts],
        g.radii[included_grid_pts],
        g.null_truth,
        grid_pt_inverse,
        g.null_hypos,
    )


# https://stackoverflow.com/a/52229385/
def hypercube_vertices(d):
    """
    The corners of a hypercube of dimension d.

    print(vertices(1))
    >>> [(1,), (-1,)]

    print(vertices(2))
    >>> [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    print(vertices(3))
    >>> [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ]

    Args:
        d: the dimension

    Returns:
        a numpy array of shape (2**d, d) containing the vertices of the
        hypercube.
    """
    return np.array(list(product((1, -1), repeat=d)))


def refine_grid(g: Grid, refine_idxs):
    """
    Refine a grid by splitting the specified grid points. We split each grid
    point in two along each dimension. The centers of the new grid points are
    offset so that the two new tiles cover the same area as the original tile.

    Note that we are not refining *tiles* here, but rather *grid points*.

    Args:
        g: the grid to refine
        refine_idxs: the indices of the grid points to refine.

    Returns:
        new_thetas: the new grid points
        new_radii: the radii for the new grid points.
        unrefined_grid: the subset of the original grid that was not refined.
        keep_tile_idxs: the indices of the tiles that were not refined.
    """
    refine_radii = g.radii[refine_idxs, None, :] * 0.5
    new_thetas = (
        g.thetas[refine_idxs, None, :]
        + hypercube_vertices(g.d)[None, :, :] * refine_radii
    ).reshape((-1, g.d))
    new_radii = np.tile(refine_radii, (1, 2**g.d, 1)).reshape((-1, g.d))

    keep_idxs = np.setdiff1d(np.arange(g.n_grid_pts), refine_idxs)
    keep_tile_idxs = np.where(np.isin(g.grid_pt_idx, keep_idxs))[0]
    return new_thetas, new_radii, keep_tile_idxs