import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

import confirm.mini_imprint.bound.binomial as ehbound
import confirm.mini_imprint.lewis_drivers as lts
from confirm.lewislib import batch
from confirm.mini_imprint import grid


@dataclass
class AdaParams:
    init_K: int
    n_K_double: int
    alpha_target: float
    grid_target: float
    bias_target: float
    nB_global: int
    nB_tile: int
    step_size: int
    tuning_min_idx: int

    @property
    def max_sim_size(self):
        return self.init_K * 2**self.n_K_double

    @property
    def sim_sizes(self):
        return self.init_K * 2 ** np.arange(0, self.n_K_double + 1)


@dataclass
class AdaData:
    unifs: np.ndarray
    unifs_order: np.ndarray
    bootstrap_idxs: Dict[int, np.ndarray]


@dataclass
class TileDB:
    data: np.ndarray
    slices: Dict[str, np.ndarray]

    @property
    def n_cols(self):
        return self.data.shape[1]

    def add_field(self, name, n_cols):
        new_slices = self.slices.copy()
        J = self.n_cols
        new_slices[name] = J if n_cols == 1 else np.s_[J : J + n_cols]
        new_data = np.concatenate(
            (self.data, np.empty((self.data.shape[0], n_cols))), axis=1
        )
        return TileDB(new_data, new_slices)

    def get(self, name):
        return self.data[:, self.slices[name]]


def empty_tiledb(n_tiles):
    return TileDB(np.empty((n_tiles, 0), dtype=np.float32), dict())


def test_tile_db():
    db = empty_tiledb(3).add_field("a", 1)
    assert db.get("a").shape == (3,)
    db.get("a")[:] = 1

    db = db.add_field("b", 2)
    db.get("b")[:] = 2
    assert db.get("a").shape == (3,)
    assert np.all(db.get("a") == 1)
    assert db.get("b").shape == (3, 2)
    assert np.all(db.get("b") == 2)


# TODO: remove
if __name__ == "__main__":
    test_tile_db()


@dataclass
class AdaState:
    g: grid.Grid
    sim_sizes: np.ndarray
    todo: np.ndarray[bool]
    db: TileDB

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError
        return self.db.get(attr)

    def refine(self, P, which_refine, null_hypos, symmetry):
        # TODO: would be nice to wrap null_hypos and symmetry into the grid
        # itself.
        refine_tile_idxs = np.where(which_refine)[0]
        refine_gridpt_idxs = self.g.grid_pt_idx[refine_tile_idxs]
        new_thetas, new_radii, keep_tile_idxs = grid.refine_grid(
            self.g, refine_gridpt_idxs
        )
        new_grid_subset = grid.build_grid(
            new_thetas,
            new_radii,
            null_hypos=null_hypos,
            symmetry_planes=symmetry,
            should_prune=True,
        )

        # NOTE: It would be possible to avoid concatenating the grid every
        # iteration. For particularly large problems, that might be a large win
        # in runtime. But the additional complexity is undesirable at the
        # moment.
        g = grid.concat_grids(grid.index_grid(self.g, keep_tile_idxs), new_grid_subset)

        sim_sizes = np.concatenate(
            [self.sim_sizes[keep_tile_idxs], np.full(new_grid_subset.n_tiles, P.init_K)]
        )
        todo = np.concatenate(
            [self.todo[keep_tile_idxs], np.ones(new_grid_subset.n_tiles, dtype=bool)]
        )
        new_db_data = np.concatenate(
            (
                self.db.data[keep_tile_idxs],
                np.empty((new_grid_subset.n_tiles, self.db.n_cols)),
            )
        )
        new_db = TileDB(new_db_data, self.db.slices)
        return AdaState(g, sim_sizes, todo, new_db)


def init_data(p, lei_obj, seed):
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)
    unifs = jax.random.uniform(
        key=key1, shape=(p.max_sim_size,) + lei_obj.unifs_shape(), dtype=jnp.float32
    )
    unifs_order = jnp.arange(0, unifs.shape[1])
    bootstrap_idxs = {
        K: jnp.concatenate(
            (
                jnp.arange(K)[None, :],
                jax.random.choice(
                    key2, K, shape=(p.nB_global + p.nB_tile, K), replace=True
                ),
            )
        ).astype(jnp.int32)
        for K in p.sim_sizes
    }
    return AdaData(unifs, unifs_order, bootstrap_idxs)


def init_state(p, g):
    sim_sizes = np.full(g.n_tiles, p.init_K)
    todo = np.ones(g.n_tiles, dtype=bool)
    tile_db = (
        empty_tiledb(g.n_tiles)
        .add_field("alpha0", 1)
        .add_field("orig_lam", 1)
        .add_field("B_lam", p.nB_global)
        .add_field("twb_min_lam", 1)
        .add_field("twb_mean_lam", 1)
        .add_field("twb_max_lam", 1)
    )
    return AdaState(g, sim_sizes, todo, tile_db)


def load(name, i):
    if i == "latest":
        # find the file with the largest checkpoint index: name/###.pkl
        available_iters = [
            int(fn[:-4]) for fn in os.listdir(name) if re.match(r"[0-9]+.pkl", fn)
        ]
        i = -1 if len(available_iters) == 0 else max(available_iters)
    if i >= 0:
        fn = f"{name}/{i}.pkl"
        print(f"loading checkpoint {fn}")
        with open(fn, "rb") as f:
            data = pickle.load(f)
        return data, i, fn
    else:
        return None, i, None


def save(fp, data):
    with open(fp, "wb") as f:
        pickle.dump(data, f)


class AdaRunner:
    def __init__(self, P, lei_obj):
        self.lei_obj = lei_obj
        self.n_arm_samples = int(lei_obj.unifs_shape()[0])

        self.grid_batch_size = (
            2**6 if jax.devices()[0].device_kind == "cpu" else 2**10
        )

        bwd_solver = ehbound.BackwardQCPSolver(n=self.n_arm_samples)

        def invert_bound(alpha, theta_0, vertices, n):
            v = vertices - theta_0
            # NOTE: OPTIMIZATION POTENTIAL: if we ever need faster EH bounds, then we
            # can only run the optimizer at a single corner. The bound is still valid
            # because we're just using a suboptimal q.

            q_opt = jax.vmap(bwd_solver.solve, in_axes=(None, 0, None))(
                theta_0, v, alpha
            )
            return jnp.min(
                jax.vmap(ehbound.q_holder_bound_bwd, in_axes=(0, None, None, 0, None))(
                    q_opt, n, theta_0, v, alpha
                )
            )

        self.batched_invert_bound = batch.batch(
            jax.jit(
                jax.vmap(invert_bound, in_axes=(None, 0, 0, None)),
                static_argnums=(0, 3),
            ),
            5 * self.grid_batch_size,
            in_axes=(None, 0, 0, None),
        )

    def step(self, P, S, D):
        S.alpha0[S.todo] = self.batched_invert_bound(
            P.alpha_target,
            S.g.theta_tiles[S.todo],
            S.g.vertices(S.todo),
            self.n_arm_samples,
        )

        bootstrap_cvs_todo = lts.bootstrap_tune_runner(
            self.lei_obj,
            S.sim_sizes[S.todo],
            S.alpha0[S.todo],
            S.g.theta_tiles[S.todo],
            S.g.null_truth[S.todo],
            D.unifs,
            D.bootstrap_idxs,
            D.unifs_order,
        )

        S.orig_lam[S.todo] = bootstrap_cvs_todo[:, 0]
        S.B_lam[S.todo] = bootstrap_cvs_todo[:, 1 : 1 + P.nB_global]

        twb_lam = bootstrap_cvs_todo[:, 1 + P.nB_global :]
        S.twb_min_lam[S.todo] = twb_lam.min(axis=1)
        S.twb_mean_lam[S.todo] = twb_lam.mean(axis=1)
        S.twb_max_lam[S.todo] = twb_lam.max(axis=1)
        S.todo[:] = False
