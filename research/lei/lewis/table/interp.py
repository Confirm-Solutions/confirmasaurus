from lewis.table.base import BaseTable

from outlaw.interp import interpn


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
