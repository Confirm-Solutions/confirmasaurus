from functools import partial

import jax
import jax.numpy as jnp


class CartesianBatcher:
    def __init__(
        self,
        lower,
        upper,
        n_batches,
        batch_size,
        n_arms,
        dtype=jnp.float64,
    ):
        n_points = n_batches * batch_size
        self.grid_1d = jnp.linspace(
            start=lower,
            stop=upper,
            num=n_points,
            dtype=dtype,
        ).reshape((n_batches, batch_size))

        batch_index = jnp.arange(0, n_batches)
        coords = jnp.meshgrid(
            *jnp.full((n_arms, batch_index.shape[-1]), batch_index),
            indexing="ij",
        )
        self.indices = jnp.concatenate(
            [c.flatten().reshape(-1, 1) for c in coords], axis=1
        )

    @partial(jax.jit, static_argnums=(0,))
    def batch(self, batch_index):
        coords = jnp.meshgrid(
            *self.grid_1d[batch_index],
            indexing="ij",
        )
        return jnp.concatenate(
            [c.flatten().reshape(-1, 1) for c in coords],
            axis=1,
        )

    def batch_indices(self):
        return self.indices
