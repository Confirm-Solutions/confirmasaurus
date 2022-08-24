import jax.numpy as jnp


class LinearInterpTable:
    def __init__(self, n_sizes, grids, tables):
        """
        Parameters:
        -----------
        n_sizes:    a 2-D array of shape (n, d).
        grid:       an n-length sequence of 2-D arrays each of shape (d, a).
        tables:     an n-length sequence of 2-D arrays each of shape (d^a, ...).
        """
        grid = jnp.array(grids)
        table = jnp.array(tables)

        self.tables = tuple(
            jnp.row_stack((sub_tables[i] for i in hashes_order))
            for sub_tables in tables
        )
        self.sizes = sizes[1:]
