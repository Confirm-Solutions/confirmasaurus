import warnings
from dataclasses import dataclass
from itertools import product
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

# TODO: corners could be a sparse matrix. that would reduce memory pressure.
# TODO: better comments on the splitting code


@dataclass
class HyperPlane:
    """A plane defined by:
    x \cdot n - c = 0
    """

    n: np.ndarray
    c: float


@dataclass
class Grid:
    """
    The first two arrays define the grid points/cells:
    - thetas: the center of each hyperrectangle.
    - radii: the half-width of each hyperrectangle in each dimension.
        (NOTE: we could rename this since it's sort of a lie.)

    The next four arrays define the tiles:
    - vertices contains the vertices of each tiles. After splitting, tiles
      may have differing numbers of vertices. The vertices array will be
      shaped: (n_tiles, max_n_vertices, n_params). For tiles that have fewer
      than max_n_vertices, the unused entries will be filled with nans.
    - grid_pt_idx is an array with an entry for each tile that contains to
      index of the original grid point from which that tile was created
    - is_regular indicates whether each tile has ever been split. Tiles that
      have been split are considered "irregular" and tiles that have never been
      split are considered "regular".
    - null_truth indicates the truth of each null hypothesis for each tile.
    """

    thetas: np.ndarray
    radii: np.ndarray
    vertices: np.ndarray
    is_regular: np.ndarray
    null_truth: np.ndarray
    grid_pt_idx: np.ndarray

    @property
    def n_tiles(self):
        return self.vertices.shape[0]

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
    return Grid(
        g.thetas,
        g.radii,
        g.vertices[idxs],
        g.is_regular[idxs],
        g.null_truth[idxs],
        g.grid_pt_idx[idxs],
    )


def concat_grids(*gs: List[Grid], shared_theta=False):
    if len(gs) == 1:
        return gs[0]

    vs = [g.vertices for g in gs]
    max_n_vertices = max([varr.shape[1] for varr in vs])
    for i, varr in enumerate(vs):
        if max_n_vertices > varr.shape[1]:
            vs[i] = np.pad(
                varr,
                ((0, 0), (0, max_n_vertices - varr.shape[1]), (0, 0)),
                constant_values=np.nan,
            )

    vertices = np.concatenate((vs), axis=0)
    is_regular = np.concatenate([g.is_regular for g in gs], axis=0)
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
    return Grid(thetas, radii, vertices, is_regular, null_truth, grid_pt_idx)


def plot_grid2d(g: Grid, null_hypos: List[HyperPlane] = []):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    polys = []
    for i in range(g.n_tiles):
        vs = g.vertices[i]
        vs = vs[~np.isnan(vs).any(axis=1)]
        centroid = np.mean(vs, axis=0)
        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        order = np.argsort(angles)
        polys.append(mpl.patches.Polygon(vs[order], fill=None, edgecolor="k"))
        plt.text(*centroid, str(i))

    plt.gca().add_collection(
        mpl.collections.PatchCollection(polys, match_original=True)
    )

    maxvs = np.max(np.where(np.isnan(g.vertices), -np.inf, g.vertices), axis=(0, 1))
    minvs = np.min(np.where(np.isnan(g.vertices), np.inf, g.vertices), axis=(0, 1))
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

    plt.show()


eps = 1e-15


@jax.jit
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


def intersect_grid(g_in: Grid, null_hypos: List[HyperPlane]):
    n_grid_pts, n_params = g_in.thetas.shape

    Hns = np.array([H.n for H in null_hypos])
    Hcs = np.array([H.c for H in null_hypos])

    gridpt_dist = g_in.thetas.dot(Hns.T) - Hcs[None]
    gridpt_any_intersect = np.any(np.abs(gridpt_dist) < g_in.radii, axis=-1)
    any_intersect = gridpt_any_intersect[g_in.grid_pt_idx]
    no_intersect = ~any_intersect
    tile_rough_dist = gridpt_dist[g_in.grid_pt_idx]
    null_truth = no_intersect[..., None] & (tile_rough_dist >= 0)

    precise_any_intersect, precise_null_truth = _precise_check_for_intersections(
        g_in.vertices[any_intersect], Hns, Hcs
    )
    null_truth[any_intersect] = precise_null_truth
    any_intersect[any_intersect] = precise_any_intersect

    g_in.null_truth = np.concatenate((g_in.null_truth, null_truth), axis=1)
    g_ignore = index_grid(g_in, ~any_intersect)
    g = index_grid(g_in, any_intersect)

    if g.n_tiles == 0:
        return g_ignore

    for iH, H in enumerate(null_hypos):
        orig_max_v_count = g.vertices.shape[1]

        # Measure the distance of each vertex from the null hypo boundary
        # it's important to allow nan dist because some tiles may not have
        # every vertex slot filled. unused vertex slots will contain nans.
        dist = g.vertices.dot(H.n) - H.c

        is_null = ((dist >= 0) | np.isnan(dist)).all(axis=1)

        # 0 means alt true, 1 means null true
        g.null_truth[is_null, iH] = 1
        g.null_truth[~is_null, iH] = 0

        # Identify the tiles to be split by checking if all the tile vertices
        # are on the same side of the plane. Give some floating point slack
        # around zero so we don't suffer from imprecision.
        to_split_or_copy = ~(
            ((dist >= -eps) | np.isnan(dist)).all(axis=1)
            | ((dist <= eps) | np.isnan(dist)).all(axis=1)
        )
        to_split = to_split_or_copy & g.is_regular
        to_copy = to_split_or_copy & ~g.is_regular

        split_idxs = np.where(to_split)[0]
        copy_idxs = np.where(to_copy)[0]

        g_keep = index_grid(g, ~to_split_or_copy)

        if copy_idxs.shape[0] == 0:
            g_copy = index_grid(g, np.s_[0:0])
        else:
            copy_null_truth = np.repeat(g.null_truth[copy_idxs], 2, axis=0)
            copy_null_truth[::2, iH] = 1
            copy_null_truth[1::2, iH] = 0
            g_copy = Grid(
                g.thetas,
                g.radii,
                np.repeat(g.vertices[copy_idxs], 2, axis=0),
                np.repeat(g.is_regular[copy_idxs], 2, axis=0),
                copy_null_truth,
                np.repeat(g.grid_pt_idx[copy_idxs], 2, axis=0),
            )

        # If we're not splitting or copying any tiles, we can move on! We're done here.
        if split_idxs.shape[0] == 0:
            g_split = index_grid(g, np.s_[0:0])
        else:
            # Intersect every tile edge with the hyperplane to find the new vertices.
            split_grid_pt_idx = g.grid_pt_idx[split_idxs]
            split_edges = get_edges(
                g.thetas[split_grid_pt_idx], g.radii[split_grid_pt_idx]
            )
            # The first n_params columns of split_edges are the vertices from which
            # the edge originates and the second n_params are the edge vector.
            split_vs = split_edges[..., :n_params]
            split_dir = split_edges[..., n_params:]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Intersect each edge with the plane.
                alpha = (H.c - split_vs.dot(H.n)) / (split_dir.dot(H.n))
                # Now we need to identify the new tile vertices. We have three
                # possible cases here:
                # 1. Intersection: indicated by 0 < alpha < 1. We give a little
                #    eps slack to ignore intersections for null planes that just barely
                #    touch a corner of a tile. In this case, we
                # 2. Non-intersection indicated by alpha not in [0, 1]. In this
                #    case, the new vertex will just be marked nan to be filtered out
                #    later.
                # 3. Non-finite alpha which also indicates no intersection. Again,
                #    we produced a nan vertex to filter out later.
                new_vs = split_vs + alpha[:, :, None] * split_dir
                new_vs = np.where(
                    (np.isfinite(new_vs))
                    & ((alpha > eps) & (alpha < 1 - eps))[..., None],
                    new_vs,
                    np.nan,
                )

            # Create the array for the new vertices. We expand in both dimensions:
            # 1. We create a new row for each tile that is being split using np.repeat.
            # 2. We create a new column for each potential additional vertex from
            #    the intersection operation above using np.concatenate. This is far
            #    more new vertices than necessary, but facilitates a nice vectorized
            #    implementation.. We will just filter out the unnecessary slots later.
            split_vertices = np.repeat(g.vertices[split_idxs], 2, axis=0)
            split_vertices = np.concatenate(
                (
                    split_vertices,
                    np.full(
                        (split_vertices.shape[0], split_edges.shape[1], n_params),
                        np.nan,
                    ),
                ),
                axis=1,
            )

            # Now we need to fill in the new vertices:
            # For each original tile vertex, we need to determine whether the tile
            # lies in the new null tile or the new alt tile.
            include_in_null_tile = dist[split_idxs] >= -eps
            include_in_alt_tile = dist[split_idxs] <= eps

            # Since we copied the entire tiles, we can "delete" vertices by
            # multiply by nan
            # note: ::2 traverses the range of new null hypo tiles
            #       1::2 traverses the range of new alt hypo tiles
            split_vertices[::2, :orig_max_v_count] *= np.where(
                include_in_null_tile, 1, np.nan
            )[..., None]
            split_vertices[1::2, :orig_max_v_count] *= np.where(
                include_in_alt_tile, 1, np.nan
            )[..., None]

            # The intersection vertices get added to both new tiles.
            split_vertices[::2, orig_max_v_count:] = new_vs
            split_vertices[1::2, orig_max_v_count:] = new_vs

            # Trim the new tile array:
            # We now are left with an array of tile vertices that has many more
            # vertex slots per tile than necessary with the unused slots filled
            # with nan.
            # To deal with this:
            # 1. We sort along the vertices axis. This has the effect of
            #    moving all the nan vertices to the end of the list.
            split_vertices = split_vertices[
                np.arange(split_vertices.shape[0])[:, None],
                np.argsort(np.sum(split_vertices, axis=-1), axis=-1),
            ]
            # 2. Identify the maximum number of vertices of any tile and trim the
            #    array so that is the new vertex dimension size
            nonfinite_corners = (~np.isfinite(split_vertices)).all(axis=(0, 2))
            # 3. If any corner is unused for all tiles, we should remove it.
            #    But, we can't trim smaller than the original vertices array.
            if nonfinite_corners[-1]:
                first_all_nan_corner = nonfinite_corners.argmax()
                split_vertices = split_vertices[:, :first_all_nan_corner]

            # Update the remaining tile characteristics.
            split_null_truth = np.repeat(g.null_truth[split_idxs], 2, axis=0)
            # - the two sides of a split tile have their null hypo truth
            # indicators updated.
            split_null_truth[::2, iH] = 1
            split_null_truth[1::2, iH] = 0
            g_split = Grid(
                g.thetas,
                g.radii,
                split_vertices,
                np.full(split_idxs.shape[0] * 2, False, dtype=bool),
                split_null_truth,
                np.repeat(split_grid_pt_idx, 2, axis=0),
            )

        # Hurray, we made it! We can concatenate our grids!
        g = concat_grids(g_keep, g_copy, g_split, shared_theta=True)
    return concat_grids(g_ignore, g, shared_theta=True)


def build_grid(
    thetas: np.ndarray, radii: np.ndarray, null_hypos: List[HyperPlane] = []
):
    """
    Construct an Imprint grid from a set of grid point centers, radii and null
    hypothesis.
    1. Initially, we construct simple hyperrectangle cells.
    2. Then, we split cells that are intersected by the null hypothesis boundaries.

    Note that we do not split cells twice. This is a simplification that makes
    the software much simpler and probably doesn't cost us much in terms of
    bound tightness because very few cells are intersected by multiple
    hyperplanes.

    Parameters
    ----------
    thetas
        The centers of the hyperrectangle grid.
    radii
        The half-width of each hyperrectangle in each dimension.
    null_hypos
        A list of hyperplanes defining the boundary of the null hypothesis. The
        normal vector of these hyperplanes point into the null domain.


    Returns
    -------
        a Grid object
    """
    n_grid_pts, n_params = thetas.shape

    # For splitting cells, we will need to know the nD edges of each cell and
    # the vertices of each tile.
    unit_vs = hypercube_vertices(n_params)
    tile_vs = thetas[:, None, :] + (unit_vs[None, :, :] * radii[:, None, :])

    # Keep track of the various tile properties. See the Grid class docstring
    # for definitions.
    grid_pt_idx = np.arange(n_grid_pts)
    is_regular = np.ones(n_grid_pts, dtype=bool)
    null_truth = np.full((n_grid_pts, 0), -1)
    g = Grid(thetas, radii, tile_vs, is_regular, null_truth, grid_pt_idx)
    if len(null_hypos) > 0:
        g = intersect_grid(g, null_hypos)
    return g


def cartesian_gridpts(theta_min, theta_max, n_theta_1d):
    """
    _summary_

    Args:
        theta_min: _description_
        theta_max: _description_
        n_theta_1d: _description_
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


def prune(g):
    """Remove tiles that are entirely within the alternative hypothesis space.

    Parameters
    ----------
    g
        the Grid object

    Returns
    -------
        the pruned Grid object.
    """
    if g.null_truth.shape[1] == 0:
        return g
    all_alt = (g.null_truth == 0).all(axis=1)
    grid_pt_idx = g.grid_pt_idx[~all_alt]
    included_grid_pts, grid_pt_inverse = np.unique(grid_pt_idx, return_inverse=True)
    return Grid(
        g.thetas[included_grid_pts],
        g.radii[included_grid_pts],
        g.vertices[~all_alt],
        g.is_regular[~all_alt],
        g.null_truth[~all_alt],
        grid_pt_inverse,
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
    """
    return np.array(list(product((1, -1), repeat=d)))


def get_edges(thetas, radii):
    """
    Construct an array indicating the edges of each hyperrectangle.
    - edges[:, :, :n_params] are the vertices at the origin of the edges
    - edges[:, :, n_params:] are the edge vectors pointing from the start to
        the end of the edge

    In total, the edges array has shape:
    (n_grid_pts, number of hypercube vertices, 2*n_params)
    """

    n_params = thetas.shape[1]
    unit_vs = hypercube_vertices(n_params)
    n_vs = unit_vs.shape[0]
    unit_edges = []
    for i in range(n_vs):
        for j in range(n_params):
            if unit_vs[i, j] > 0:
                continue
            unit_edges.append(np.concatenate((unit_vs[i], np.identity(n_params)[j])))

    edges = np.tile(np.array(unit_edges)[None, :, :], (thetas.shape[0], 1, 1))
    edges[:, :, :n_params] *= radii[:, None, :]
    edges[:, :, n_params:] *= 2 * radii[:, None, :]
    edges[:, :, :n_params] += thetas[:, None, :]
    return edges


def refine_grid(g: Grid, refine_idxs: np.ndarray[int]):
    refine_radii = g.radii[refine_idxs, None, :] * 0.5
    new_thetas = (
        g.thetas[refine_idxs, None, :]
        + hypercube_vertices(g.d)[None, :, :] * refine_radii
    ).reshape((-1, g.d))
    new_radii = np.tile(refine_radii, (1, 2**g.d, 1)).reshape((-1, g.d))

    keep_idxs = np.setdiff1d(np.arange(g.n_grid_pts), refine_idxs)
    keep_tile_idxs = np.where(np.isin(g.grid_pt_idx, keep_idxs))[0]
    _, keep_grid_pt_inverse = np.unique(
        g.grid_pt_idx[keep_tile_idxs], return_inverse=True
    )
    unrefined_grid = Grid(
        g.thetas[keep_idxs],
        g.radii[keep_idxs],
        g.vertices[keep_tile_idxs],
        g.is_regular[keep_tile_idxs],
        g.null_truth[keep_tile_idxs],
        keep_grid_pt_inverse,
    )

    return new_thetas, new_radii, unrefined_grid, keep_tile_idxs
