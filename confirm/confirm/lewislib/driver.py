import time

import jax
import numpy as np

from . import batch


class LeiSimulator:
    def __init__(
        self,
        lei_obj,
        p_tiles,
        null_truths,
        grid_batch_size,
        reduce_func=None,
    ):
        self.lei_obj = lei_obj
        self.unifs_shape = self.lei_obj.unifs_shape()
        self.unifs_order = np.arange(0, self.unifs_shape[0])
        self.p_tiles = p_tiles
        self.null_truths = null_truths
        self.grid_batch_size = grid_batch_size

        self.reduce_func = (
            lambda x: np.sum(x, axis=0) if not reduce_func else reduce_func
        )

        self.f_batch_sim_batch_grid_jit = jax.jit(self.f_batch_sim_batch_grid)
        self.batch_all = batch.batch_all(
            self.f_batch_sim_batch_grid_jit,
            batch_size=self.grid_batch_size,
            in_axes=(0, 0, None, None),
        )

        self.typeI_sum = None
        self.typeI_score = None

    def f_batch_sim_batch_grid(self, p_batch, null_batch, unifs_batch, unifs_order):
        return jax.vmap(
            jax.vmap(
                self.lei_obj.simulate_rejection,
                in_axes=(0, 0, None, None),
            ),
            in_axes=(None, None, 0, None),
        )(p_batch, null_batch, unifs_batch, unifs_order)

    def simulate_batch_sim(self, sim_batch_size, i, key):
        start = time.perf_counter()

        unifs = jax.random.uniform(key=key, shape=(sim_batch_size,) + self.unifs_shape)
        rejs_scores, n_padded = self.batch_all(
            self.p_tiles, self.null_truths, unifs, self.unifs_order
        )
        rejs, scores = tuple(
            np.concatenate(
                tuple(x[i] for x in rejs_scores),
                axis=1,
            )
            for i in range(2)
        )
        rejs, scores = (
            (rejs[:, :-n_padded], scores[:, :-n_padded, :])
            if n_padded
            else (rejs, scores)
        )
        rejs_reduced = self.reduce_func(rejs)
        scores_reduced = self.reduce_func(scores)

        end = time.perf_counter()
        elapsed_time = end - start
        print(f"Batch {i}: {elapsed_time:.03f}s")
        return rejs_reduced, scores_reduced

    def simulate(
        self,
        key,
        n_sim_batches,
        sim_batch_size,
    ):
        keys = jax.random.split(key, num=n_sim_batches)
        self.typeI_sum = np.zeros(self.p_tiles.shape[0])
        self.typeI_score = np.zeros(self.p_tiles.shape)
        for i, key in enumerate(keys):
            out = self.simulate_batch_sim(sim_batch_size, i, key)
            self.typeI_sum += out[0]
            self.typeI_score += out[1]
        return self.typeI_sum, self.typeI_score
