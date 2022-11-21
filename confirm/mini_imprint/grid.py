import time
import warnings
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import List

import numpy as np
import pandas as pd
import sympy as sp


@dataclass(eq=False)
class HyperPlane:
    """
    A plane defined by:
    x.dot(n) - c = 0

    Sign convention: When used as the boundary between null hypothesis and
    alternative, the normal should point towards the null hypothesis space.
    """

    n: np.ndarray
    c: float

    def __eq__(self, other):
        if not isinstance(other, HyperPlane):
            return NotImplemented
        return np.allclose(self.n, other.n) and np.isclose(self.c, other.c)


def hypo(str_expr):
    alias = dict(
        x="x0",
        y="x1",
        z="x2",
    )
    expr = sp.parsing.parse_expr(str_expr)
    if isinstance(expr, sp.StrictLessThan) or isinstance(expr, sp.LessThan):
        plane = expr.rhs - expr.lhs
    elif isinstance(expr, sp.StrictGreaterThan) or isinstance(expr, sp.GreaterThan):
        plane = expr.lhs - expr.rhs
    else:
        raise ValueError("Hypothesis expression must be an inequality.")

    symbols = plane.free_symbols
    coeffs = sp.Poly(plane, *symbols).coeffs()
    if len(coeffs) > len(symbols):
        c = -float(coeffs[-1])
        coeffs = coeffs[:-1]
    else:
        c = 0

    symbol_names = [alias.get(s.name, s.name).replace("theta", "x") for s in symbols]

    if any([s[0] != "x" for s in symbol_names]):
        raise ValueError(
            f"Hypothesis contains invalid symbols: {symbols}."
            " Valid symbols are x0..., theta0..., x, y, z."
        )
    try:
        symbol_idxs = [int(s[1:]) for s in symbol_names]
    except ValueError:
        raise ValueError(
            f"Hypothesis contains invalid symbols: {symbols}."
            " Valid symbols are x0..., theta0..., x, y, z."
        )
    coeff_dict = dict(zip(symbol_idxs, coeffs))
    max_idx = max(symbol_idxs)

    n = [float(coeff_dict.get(i, 0)) for i in range(max_idx + 1)]
    n_norm = np.linalg.norm(n)
    n /= n_norm
    c /= n_norm

    return HyperPlane(np.array(n), c)


@dataclass
class Grid:
    df: pd.DataFrame
    null_hypos: List[HyperPlane] = field(default_factory=lambda: [])

    @property
    def d(self):
        if not hasattr(self, "_d"):
            self._d = (
                max([int(c[5:]) for c in self.df.columns if c.startswith("theta")]) + 1
            )
        return self._d

    @property
    def n_tiles(self):
        return self.df.shape[0]

    @property
    def n_active_tiles(self):
        return self.df["active"].sum()

    def _add_null_hypo(self, H):
        eps = 1e-15

        hypo_idx = len(self.null_hypos)
        self.null_hypos.append(H)

        theta, vertices = self.get_theta_and_vertices()
        radii = self.get_radii()

        gridpt_dist = theta.dot(H.n) - H.c
        self.df[f"null_truth{hypo_idx}"] = gridpt_dist >= 0

        close = np.abs(gridpt_dist) <= np.sqrt(np.sum(self.get_radii() ** 2, axis=-1))
        # Ignore intersections of inactive tiles.
        close &= self.df["active"].values

        vertex_dist = vertices[close].dot(H.n) - H.c
        all_above = (vertex_dist >= -eps).all(axis=-1)
        all_below = (vertex_dist <= eps).all(axis=-1)
        close_intersects = ~(all_above | all_below)
        if close_intersects.sum() == 0:
            return self

        intersects = np.zeros(self.n_tiles, dtype=bool)
        intersects[close] = close_intersects

        new_theta, new_radii = split(
            theta[intersects],
            radii[intersects],
            vertices[intersects],
            vertex_dist[close_intersects],
            H,
        )

        parent_K = np.repeat(self.df["K"].values[intersects], 2)
        parent_id = np.repeat(self.df["id"].values[intersects], 2)
        birthday = np.repeat(self.df["birthday"].values[intersects], 2)
        new_g = init_grid(
            new_theta, new_radii, parent_K, parents=parent_id, birthday=birthday
        )
        for i in range(hypo_idx):
            new_g.df[f"null_truth{i}"] = np.repeat(
                self.df[f"null_truth{i}"].values[intersects], 2
            )
        new_g.df[f"null_truth{hypo_idx}"] = True
        new_g.df[f"null_truth{hypo_idx}"].values[1::2] = False

        self.df["active"].values[intersects] = False
        self.df["eligible"].values[intersects] = False
        return self.concat(new_g)

    def add_null_hypos(self, null_hypos):
        g = Grid(self.df.copy(), self.null_hypos)
        for H in null_hypos:
            Hn = np.asarray(H.n)
            Hpad = HyperPlane(np.pad(Hn, (0, g.d - Hn.shape[0])), H.c)
            g = g._add_null_hypo(Hpad)
        return g

    def prune(self):
        if len(self.null_hypos) == 0:
            return self
        null_truth = self.get_null_truth()
        which = (null_truth.any(axis=1)) | (null_truth.shape[1] == 0)
        if np.all(which):
            return self
        return self.subset(which)

    def add_cols(self, df):
        return Grid(pd.concat((self.df, df), axis=1), self.null_hypos)

    def subset(self, which):
        df = self.df.loc[which].reset_index(drop=True)
        return Grid(df, self.null_hypos)

    def active(self):
        return self.subset(self.df["active"])

    def get_null_truth(self):
        return self.df[
            [
                f"null_truth{i}"
                for i in range(self.df.shape[1])
                if f"null_truth{i}" in self.df.columns
            ]
        ].to_numpy()

    def get_theta(self):
        return self.df[[f"theta{i}" for i in range(self.d)]].to_numpy()

    def get_radii(self):
        return self.df[[f"radii{i}" for i in range(self.d)]].to_numpy()

    def get_theta_and_vertices(self):
        theta = self.get_theta()
        return theta, (
            theta[:, None, :]
            + hypercube_vertices(self.d)[None, :, :] * self.get_radii()[:, None, :]
        )

    def refine(self):
        refine_radii = self.get_radii()[:, None, :] * 0.5
        refine_theta = self.get_theta()[:, None, :]
        new_thetas = (
            refine_theta + hypercube_vertices(self.d)[None, :, :] * refine_radii
        ).reshape((-1, self.d))
        new_radii = np.tile(refine_radii, (1, 2**self.d, 1)).reshape((-1, self.d))
        parent_K = np.repeat(self.df["K"].values, 2**self.d)
        parent_id = np.repeat(self.df["id"].values, 2**self.d)
        return init_grid(
            new_thetas,
            new_radii,
            parent_K,
            parents=parent_id,
        )

    def concat(self, other):
        return Grid(
            pd.concat((self.df, other.df), axis=0, ignore_index=True), self.null_hypos
        )


def init_grid(theta, radii, init_K=None, parents=None, birthday=0):
    d = theta.shape[1]
    indict = dict()
    for i in range(d):
        indict[f"theta{i}"] = theta[:, i]
    for i in range(d):
        indict[f"radii{i}"] = radii[:, i]

    indict["K"] = init_K if init_K is not None else 0
    indict["parent_id"] = (
        parents.astype(np.uint64) if parents is not None else np.uint64(0)
    )
    indict["birthday"] = birthday

    # Is this a terminal node in the tree?
    indict["active"] = True
    # Is this node currently being processed?
    indict["locked"] = False
    # Is this node eligible for processing?
    indict["eligible"] = True

    indict["id"] = gen_short_uuids(len(theta))

    return Grid(pd.DataFrame(indict), [])


def cartesian_grid(theta_min, theta_max, *, n=None, null_hypos=None, prune=True):
    theta_min = np.asarray(theta_min)
    theta_max = np.asarray(theta_max)

    if n is None:
        n = np.full(theta_min.shape[0], 2)
    g = init_grid(*_cartesian_gridpts(theta_min, theta_max, n))
    if null_hypos is not None:
        g = g.add_null_hypos(null_hypos)
        if prune:
            g = g.prune()
    return g


def _cartesian_gridpts(theta_min, theta_max, n_theta_1d):
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
    radii1d = [
        np.full(
            theta1d[i].shape[0], (theta_max[i] - theta_min[i]) / (2 * n_theta_1d[i])
        )
        for i in range(n_arms)
    ]
    theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
    radii = np.stack(np.meshgrid(*radii1d), axis=-1).reshape((-1, len(theta1d)))
    return theta, radii


def plot_grid(g: Grid, only_active=True, dims=(0, 1)):
    """
    Plot a 2D grid.

    Args:
        g: the grid
        null_hypos: If provided, the function will plot red lines for the null
            hypothesis boundaries. Defaults to [].
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    vertices = g.get_theta_and_vertices()[1][..., dims]

    if only_active:
        g = g.active()

    polys = []
    for i in range(vertices.shape[0]):
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

    maxvs = np.max(vertices, axis=(0, 1))
    minvs = np.min(vertices, axis=(0, 1))
    view_center = 0.5 * (maxvs + minvs)
    view_radius = (maxvs - minvs) * 0.55
    xlims = view_center[0] + np.array([-1, 1]) * view_radius[0]
    ylims = view_center[1] + np.array([-1, 1]) * view_radius[1]
    plt.xlim(xlims)
    plt.ylim(ylims)

    for h in g.null_hypos:
        if h.n[0] == 0:
            xs = np.linspace(*xlims, 100)
            ys = (h.c - xs * h.n[0]) / h.n[1]
        else:
            ys = np.linspace(*ylims, 100)
            xs = (h.c - ys * h.n[1]) / h.n[0]
        plt.plot(xs, ys, "r-")


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
    return np.array(list(product((-1, 1), repeat=d)))


def split(theta, radii, vertices, vertex_dist, H):
    eps = 1e-15
    d = theta.shape[1]

    ########################################
    # Step 1. Intersect tile edges with the hyperplane.
    # This will identify the new vertices that we need to add.
    ########################################
    split_edges = get_edges(theta, radii)
    # The first n_params columns of split_edges are the vertices from which
    # the edge originates and the second n_params are the edge vector.
    split_vs = split_edges[..., :d]
    split_dir = split_edges[..., d:]

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
            (np.isfinite(new_vs)) & ((alpha > eps) & (alpha < 1 - eps))[..., None],
            new_vs,
            np.nan,
        )

    ########################################
    # Step 2. Construct the vertex array for the new tiles..
    ########################################
    # Create the array for the new vertices. We need to expand the
    # original vertex array in both dimensions:
    # 1. We create a new row for each tile that is being split using np.repeat.
    # 2. We create a new column for each potential additional vertex from
    #    the intersection operation above using np.concatenate. This is
    #    more new vertices than necessary, but facilitates a nice
    #    vectorized implementation.. We will just filter out the
    #    unnecessary slots later.
    split_vertices = np.repeat(vertices, 2, axis=0)
    split_vertices = np.concatenate(
        (
            split_vertices,
            np.full(
                (split_vertices.shape[0], split_edges.shape[1], d),
                np.nan,
            ),
        ),
        axis=1,
    )

    # Now we need to fill in the new vertices:
    # For each original tile vertex, we need to determine whether the tile
    # lies in the new null tile or the new alt tile.
    include_in_null_tile = vertex_dist >= -eps
    include_in_alt_tile = vertex_dist <= eps

    # Since we copied the entire tiles, we can "delete" vertices by
    # multiply by nan
    # note: ::2 traverses the range of new null hypo tiles
    #       1::2 traverses the range of new alt hypo tiles
    split_vertices[::2, : vertices.shape[1]] *= np.where(
        include_in_null_tile, 1, np.nan
    )[..., None]
    split_vertices[1::2, : vertices.shape[1]] *= np.where(
        include_in_alt_tile, 1, np.nan
    )[..., None]

    # The intersection vertices get added to both new tiles because
    # they lie on the boundary between the two tiles.
    split_vertices[::2, vertices.shape[1] :] = new_vs
    split_vertices[1::2, vertices.shape[1] :] = new_vs

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

    ########################################
    # Step 3. Identify bounding boxes.
    ########################################
    min_val = np.nanmin(split_vertices, axis=1)
    max_val = np.nanmax(split_vertices, axis=1)
    new_theta = (min_val + max_val) / 2
    new_radii = (max_val - min_val) / 2
    return new_theta, new_radii


def get_edges(theta, radii):
    """
    Construct an array indicating the edges of each hyperrectangle.
    - edges[:, :, :n_params] are the vertices at the origin of the edges
    - edges[:, :, n_params:] are the edge vectors pointing from the start to
        the end of the edge

    Args:
        thetas: the centers of the hyperrectangles
        radii: the half-width of the hyperrectangles

    Returns:
        edges: an array as specified in the docstring shaped like
             (n_grid_pts, number of hypercube vertices, 2*n_params)
    """

    n_params = theta.shape[1]
    unit_vs = hypercube_vertices(n_params)
    n_vs = unit_vs.shape[0]
    unit_edges = []
    for i in range(n_vs):
        for j in range(n_params):
            if unit_vs[i, j] > 0:
                continue
            unit_edges.append(np.concatenate((unit_vs[i], np.identity(n_params)[j])))

    edges = np.tile(np.array(unit_edges)[None, :, :], (theta.shape[0], 1, 1))
    edges[:, :, :n_params] *= radii[:, None, :]
    edges[:, :, n_params:] *= 2 * radii[:, None, :]
    edges[:, :, :n_params] += theta[:, None, :]
    return edges


def gen_short_uuids(n, host_id=None, t=None):
    """
    Short UUIDs are a custom identifier created for imprint that should allow
    for concurrent creation of tiles without having overlapping indices.

    - The lowest 20 bits are the index of the created tiles within this batch.
      This allows for up to 2^20 = ~1 million tiles to be created in a single
      batch. This is not a problematic constraint, because we can just call the
      function again for more IDs.
    - The next 14 bits are the index of the process. This is a pretty generous limit
      on the number of processes since 2^14=16384.
    - The highest 30 bits are the time in seconds of creation. This will not
      loop for 34 years. When we start running jobs that take longer than 34
      years to complete, please send a message to me in the afterlife.
        - The creation time is never re-used. If the creation time is going to
          be reused because less than one second has passed since the previous
          call to gen_short_uuids, then the creation time is incremented by
          one.

    NOTE: This should be safe across processes but will not be safe across
    threads within a single Python process because multithreaded programs share
    globals.

    Args:
        n: The number of short uuids to generate.
        host_id: The host id. It's okay to ignore this for non-concurrent jobs.
            Defaults to None.
        t: The time to impose (used for testing). Defaults to None.

    Returns:
        An array with dtype uint64 of length n containing short uuids.
    """
    n_max = 2 ** _gen_short_uuids.config[0] - 1
    if n <= n_max:
        return _gen_short_uuids(n, host_id, t)

    out = np.empty(n, dtype=np.uint64)
    for i in range(0, n, n_max):
        chunk_size = min(n_max, n - i)
        out[i : i + chunk_size] = _gen_short_uuids(chunk_size, host_id, t)
    return out


def _gen_short_uuids(n, host_id, t):
    n_bits, host_bits = _gen_short_uuids.config
    # time_bits = 64 - n_bits - host_bits
    assert n < 2**n_bits

    if host_id is None:
        # host_id == 0 is skipped so that we can use 0 as a sentinel value
        host_id = 1
    assert host_id > 0
    assert host_id < 2**host_bits

    if t is None:
        t = np.uint64(int(time.time()))
    if _gen_short_uuids.largest_t is not None and t <= _gen_short_uuids.largest_t:
        t = np.uint64(_gen_short_uuids.largest_t + 1)
    _gen_short_uuids.largest_t = t

    return (
        (t << np.uint64(n_bits + host_bits))
        + np.uint64(host_id << n_bits)
        + np.arange(n, dtype=np.uint64)
    )


_gen_short_uuids.config = (20, 14)
_gen_short_uuids.largest_t = None
