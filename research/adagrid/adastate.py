import os
import pickle
import re
from dataclasses import dataclass

import numpy as np

from confirm.mini_imprint import grid


@dataclass
class AdaParams:
    init_K: int
    n_sim_double: int
    alpha_target: float
    grid_target: float
    bias_target: float

    @property
    def max_sim_size(self):
        return self.init_K * 2**self.n_sim_double


@dataclass
class AdaState:
    g: grid.Grid
    sim_sizes: np.ndarray
    tile_data: np.ndarray
    # TODO: can be lumped into tile_data
    pointwise_target_alpha: np.ndarray
    # TODO: tile data accessor based on name.

    def save(self, name, iter):
        fn = f"{name}/{iter}.pkl"
        with open(fn, "wb") as f:
            pickle.dump(self, f)


def load_iter(name, iter):
    load_iter = "latest"
    if load_iter == "latest":
        # find the file with the largest checkpoint index: name/###.pkl
        available_iters = [
            int(fn[:-4]) for fn in os.listdir(name) if re.match(r"[0-9]+.pkl", fn)
        ]
        load_iter = -1 if len(available_iters) == 0 else max(available_iters)

        fn = f"{name}/{load_iter}.pkl"
        print(f"loading checkpoint {fn}")
        with open(fn, "rb") as f:
            data = pickle.load(f)
    return data, load_iter, fn


def save_state():
    pass


def init_state(init_K, nB_global, g):
    sim_sizes = np.full(g.n_tiles, init_K)
    tile_data = np.empty((g.n_tiles, 3 + nB_global), dtype=float)
    pointwise_target_alpha = np.empty(g.n_tiles, dtype=float)
    return AdaState(
        g,
        sim_sizes,
        tile_data,
        pointwise_target_alpha,
    )
