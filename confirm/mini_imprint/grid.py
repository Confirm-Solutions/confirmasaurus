from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


@dataclass
class Grid:
    d: int
    df: pd.DataFrame
    null_hypos: list

    @property
    def n_tiles(self):
        return self.df.shape[0]

    def add_null_hypo(self, null_hypo):
        # TODO: I hardcoded the null hypothesis here. We can just migrate over
        # the old grid code. It should still work great.
        self.df["null_truth0"] = self.df["theta0"] < 0
        intersects = np.abs(self.df["theta0"]) < self.df["radii0"]
        intersects.sum()
        df_copy = self.df.loc[self.df.index.repeat(intersects + 1)].reset_index(
            drop=True
        )

        which_intersects = np.where(intersects)[0]
        offset = np.cumsum(intersects)[which_intersects]
        df_copy.loc[which_intersects + offset - 1, "null_truth0"] = False
        df_copy.loc[which_intersects + offset, "null_truth0"] = True
        return Grid(self.d, df_copy, self.null_hypos + [null_hypo])

    def prune(self):
        if len(self.null_hypos) == 0:
            return self
        null_truth = self.get_null_truth()
        which = (null_truth.any(axis=1)) | (null_truth.shape[1] == 0)
        if np.all(which):
            return self
        return Grid(self.d, self.df.loc[which].copy(), self.null_hypos)

    def get_null_truth(self):
        return self.df[
            [f"null_truth{i}" for i in range(len(self.null_hypos))]
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

    def add_cols(self, df):
        return Grid(self.d, pd.concat((self.df, df), axis=1), self.null_hypos)


def init_grid(theta, radii, init_K):
    d = theta.shape[1]
    indict = dict(
        K=init_K,
    )
    for i in range(d):
        indict[f"theta{i}"] = theta[:, i]
    for i in range(d):
        indict[f"radii{i}"] = radii[:, i]
    indict["grid_pt_idx"] = np.arange(theta.shape[0])
    indict["parent_idx"] = -1
    indict["birthday"] = 0
    return Grid(d, pd.DataFrame(indict), [])


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
