import jax.numpy as jnp
import lewis.jax_wrappers as jwp

from outlaw.interp import interpn


class BaseTable:
    def __init__(self, n_sizes):
        # compute mask to hash n_sizes
        n_arms = n_sizes.shape[-1]
        n_sizes_max = jnp.max(n_sizes) + 1
        n_sizes_max_mask = n_sizes_max ** jnp.arange(0, n_arms)
        self.n_sizes_max_mask = n_sizes_max_mask.astype(int)

        # create hashes
        hashes = jnp.array([self.hash_n__(ns) for ns in n_sizes])

        # reorder data based on increasing order of hashes
        self.hashes_order = jnp.argsort(hashes)
        self.hashes = hashes[self.hashes_order]

    def hash_n__(self, n):
        """
        Hashes the n configuration with a given mask.

        Parameters:
        -----------
        n:      n configuration sorted in decreasing order.
        """
        return jnp.sum(n * self.n_sizes_max_mask)

    def search(self, n):
        n_hash = self.hash_n__(n)
        idx = jnp.searchsorted(self.hashes, n_hash)
        return idx

    def hash_ordered(self, seq):
        return tuple(seq[i] for i in self.hashes_order)


class LinearInterpTable(BaseTable):
    def __init__(self, n_sizes, grids, tables):
        """
        Parameters:
        -----------
        n_sizes:    a 2-D array of shape (n, d).
        grid:       a 3-D array of shape (n, d, a).
        tables:     a sequence of N-D arrays
                    each of shape (n, a^d, ...) where each slice
                    (a^d, ...) corresponds to values in a cartesian
                    product of points defined by the same slice of grid.
        """
        super().__init__(n_sizes)
        if not isinstance(tables, tuple):
            tables = (tables,)

        self.grids = grids[self.hashes_order]

        self.tables = tuple(sub_tables[self.hashes_order] for sub_tables in tables)

        n_arms, n_points = self.grids.shape[-2:]
        self.shape = tuple(n_points for _ in range(n_arms))

    def at(self, data):
        y = data[:, 0]
        n = data[:, 1]
        idx = self.search(n)
        grid = self.grids[idx]
        return tuple(
            interpn(grid, values[idx].reshape(self.shape + values[idx].shape[1:]), y)
            for values in self.tables
        )


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
