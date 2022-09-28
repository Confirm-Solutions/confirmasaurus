from dataclasses import dataclass

import numpy as np
import scipy.spatial

from . import binomial
from . import grid


def chunked_simulate(
    g, accumulator, sim_size, n_arm_samples, tile_chunk_size=5000, sim_chunk_size=50000
):
    n_tiles, n_params = g.theta_tiles.shape

    typeI_sum = np.zeros(n_tiles)
    typeI_score = np.zeros((n_tiles, n_params))
    n_tile_chunks = int(np.ceil(n_tiles / tile_chunk_size))
    n_sim_chunks = int(np.ceil(sim_size / sim_chunk_size))

    for j in range(n_sim_chunks):
        start_sims = j * sim_chunk_size
        end_sims = min(start_sims + sim_chunk_size, sim_size)
        nsims_this_chunk = end_sims - start_sims

        # abstraction idea: this part could be controlled by accumulator/model?
        samples = np.random.uniform(size=(nsims_this_chunk, n_arm_samples, n_params))
        for i in range(n_tile_chunks):
            gridpt_start = i * tile_chunk_size
            gridpt_end = (i + 1) * tile_chunk_size
            gridpt_end = min(gridpt_end, g.theta_tiles.shape[0])
            sum_chunk, score_chunk = accumulator(
                g.theta_tiles[gridpt_start:gridpt_end],
                g.null_truth[gridpt_start:gridpt_end],
                samples,
            )
            typeI_sum[gridpt_start:gridpt_end] += sum_chunk
            typeI_score[gridpt_start:gridpt_end] += score_chunk
    return typeI_sum, typeI_score


@dataclass
class AdaState:
    g: grid.Grid
    typeI_sum: np.ndarray
    sim_sizes: np.ndarray
    typeI_est: np.ndarray
    typeI_CI: np.ndarray
    hob_upper: np.ndarray
    target_sim_sizes: np.ndarray
    # NOTE: delta/holderq probably shouldn't live here. feels like an
    # abstraction failure.
    delta: float
    holderq: int
    total_sims: int


def ada_setup(g, n_initial_sims: int, delta: float, holderq: int):
    return AdaState(
        g,
        typeI_sum=np.zeros(g.n_tiles),
        sim_sizes=np.zeros(g.n_tiles, dtype=int),
        typeI_est=np.empty(g.n_tiles),
        typeI_CI=np.empty(g.n_tiles),
        hob_upper=np.empty(g.n_tiles),
        target_sim_sizes=np.full(g.n_tiles, n_initial_sims, dtype=int),
        delta=delta,
        holderq=holderq,
        total_sims=0,
    )


# abstraction idea: n_arm_samples could be controlled by accumulator/model?
def ada_simulate(A: AdaState, accumulator, n_arm_samples):
    updated = np.zeros(A.g.n_tiles, dtype=bool)
    while np.any(A.target_sim_sizes > A.sim_sizes):
        add_sims = A.target_sim_sizes - A.sim_sizes
        sims_to_run = np.min(add_sims[add_sims > 0])
        which_tiles = np.where(add_sims >= sims_to_run)[0]
        sim_grid = grid.index_grid(A.g, which_tiles)
        add_typeI_sum, _ = chunked_simulate(
            sim_grid, accumulator, sims_to_run, n_arm_samples
        )
        add_sim_sizes = np.full(sim_grid.n_tiles, sims_to_run)
        A.typeI_sum[which_tiles] += add_typeI_sum
        A.sim_sizes[which_tiles] += add_sim_sizes
        A.total_sims += add_sim_sizes.sum()
        updated[which_tiles] = True
    A.typeI_est[updated], A.typeI_CI[updated] = binomial.zero_order_bound(
        A.typeI_sum[updated], A.sim_sizes[updated], A.delta, 1.0
    )
    A.hob_upper[updated] = binomial.holder_odi_bound(
        A.typeI_est[updated] + A.typeI_CI[updated],
        A.g.theta_tiles[updated],
        A.g.vertices[updated],
        n_arm_samples,
        A.holderq,
    )
    return A


def ada_refine(A: AdaState, criterion):
    refine_tile_idxs = np.where(criterion)
    if refine_tile_idxs[0].shape[0] == 0:
        return A, False

    refine_gridpt_idxs = A.g.grid_pt_idx[refine_tile_idxs]
    new_thetas, new_radii, unrefined_grid, keep_tile_idxs = grid.refine_grid(
        A.g, refine_gridpt_idxs
    )
    new_grid = grid.prune(grid.build_grid(new_thetas, new_radii, A.g.null_hypos))

    full_grid = grid.concat_grids(unrefined_grid, new_grid)
    full_typeI_sum = np.concatenate(
        (A.typeI_sum[keep_tile_idxs], np.zeros(new_grid.n_tiles))
    )
    full_sim_sizes = np.concatenate(
        (A.sim_sizes[keep_tile_idxs], np.zeros(new_grid.n_tiles, dtype=int))
    )
    full_typeI_est = np.concatenate(
        (A.typeI_est[keep_tile_idxs], np.empty(new_grid.n_tiles))
    )
    full_typeI_CI = np.concatenate(
        (A.typeI_CI[keep_tile_idxs], np.empty(new_grid.n_tiles))
    )
    full_hob_upper = np.concatenate(
        (A.hob_upper[keep_tile_idxs], np.empty(new_grid.n_tiles))
    )

    # NOTE: need to start child tiles with the same target_sim_size as the
    # parent.
    # - this is tricky because i don't currently have a good way to
    #   track parents through the system.
    # - i could set up the grid structure in some sort of tree structure so
    #   that grid pts and tiles track their parents.
    # - i could just use a kdtree to find the nearest parents and then average
    #   them. this is a hacky solution but i like it! this might become a
    #   bottleneck when we have many tiles.
    nearest_parent_tiles = scipy.spatial.KDTree(A.g.theta_tiles).query(
        new_grid.theta_tiles, k=2
    )
    new_target_sim_sizes = np.mean(A.target_sim_sizes[nearest_parent_tiles[1]], axis=1)
    full_target_sim_sizes = np.concatenate(
        (A.target_sim_sizes[keep_tile_idxs], new_target_sim_sizes.astype(int))
    )
    return (
        AdaState(
            full_grid,
            full_typeI_sum,
            full_sim_sizes,
            full_typeI_est,
            full_typeI_CI,
            full_hob_upper,
            full_target_sim_sizes,
            A.delta,
            A.holderq,
            A.total_sims,
        ),
        True,
    )
