from dataclasses import dataclass

import numpy as np
import pandas as pd

from confirm.mini_imprint.grid import hypercube_vertices


@dataclass
class Grid:
    d: int
    df: pd.DataFrame
    null_hypos: list

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

    def get_theta(self, idx):
        return self.df[[f"theta{i}" for i in range(self.d)]].to_numpy()

    def get_radii(self):
        return self.df[[f"radii{i}" for i in range(self.d)]].to_numpy()

    def get_theta_and_vertices(self):
        theta = self.get_theta()
        return theta, (
            theta[:, None, :]
            + hypercube_vertices(self.d)[None, :, :] * self.get_radii()[:, None, :]
        )


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
    indict["parent_idx"] = None
    indict["birthday"] = 0
    return Grid(d, pd.DataFrame(indict), [])
