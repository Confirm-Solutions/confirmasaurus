import jax.numpy as jnp
import lewis.jax_wrappers as jwp
from lewis.table.base import BaseTable


class LookupTable(BaseTable):
    def __init__(
        self,
        n_sizes,
        tables,
    ):
        """
        Constructs a lookup table given a list of n sizes
        and their corresponding table of values corresponding to
        all enumerations of the sizes.

        Parameters:
        -----------
        n_sizes:    a 2-D array of shape (n, d) where n is the number
                    of configurations and d is the number of arms.
        tables:     a list of list of/list of/table of values.
                    If it is not a list of list of tables,
                    it will be converted in such a form.
                    In that form, tables[i] corresponds to the ith table
                    where tables[i][j] is a sub-table of values
                    corresponding to the configuration n_sizes[j].
                    tables[i][j] is assumed to be of shape
                    (jnp.prod(n_sizes[j]), ...).
                    Each row is a value corresponding to a row of
                    a d-dimensional possible configuration y, where
                    0 <= y < n_sizes, where the first index increments slowest
                    and the last index increments fastest.
        """
        super().__init__(n_sizes)

        # force tables to be a tuple of (tuple of sub-tables)
        if not isinstance(tables, tuple):
            tables = ((tables,),)
        if not isinstance(tables[0], tuple):
            tables = (tables,)

        tables_reordered = tuple(self.hash_ordered(sub_tables) for sub_tables in tables)
        self.tables = tuple(
            jnp.row_stack(sub_tables) for sub_tables in tables_reordered
        )

        # reorder based on hash order
        n_sizes = n_sizes[self.hashes_order]

        # compute offsets corresponding to each n_size
        sizes = jnp.array([0] + [jnp.prod(ns) for ns in n_sizes])
        sizes_cumsum = jnp.cumsum(sizes)
        self.offsets = sizes_cumsum[:-1]
        self.sizes = sizes[1:]

    def at(self, data):
        index = data[:, 0]
        n = data[:, 1]
        idx = self.search(n)
        offset = self.offsets[idx]
        size = self.sizes[idx]
        slices = tuple(jwp.slice0(t, offset, offset + size) for t in self.tables)
        slices_reshaped = tuple(jwp.reshape0(a, n) for a in slices)
        return tuple(a[index] for a in slices_reshaped)
