from pathlib import Path

import numpy as np
import pandas as pd

from confirm.mini_imprint import db
from confirm.mini_imprint import grid


def example_grid(x1, x2):
    N = 10
    theta, radii = grid.cartesian_gridpts([x1], [x2], [N])
    return grid.init_grid(theta, radii, 50).add_null_hypo(0).prune()


def assert_frame_equal_no_index(df1, df2):
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True), df2.reset_index(drop=True)
    )


def prepped_dbs():
    g = example_grid(-1, 1)
    pd_tiles = db.PandasTiles.create(g)
    db_tiles = db.DuckDBTiles.create(g)
    return g, pd_tiles, db_tiles


def test_create():
    g, pd_tiles, db_tiles = prepped_dbs()
    assert_frame_equal_no_index(g.df, pd_tiles.get_all())
    assert_frame_equal_no_index(pd_tiles.get_all(), db_tiles.get_all())


def test_write():
    g, pd_tiles, db_tiles = prepped_dbs()

    g2 = example_grid(-2, -1)
    pd_tiles.write(g2)
    db_tiles.write(g2)

    assert_frame_equal_no_index(pd_tiles.get_all(), db_tiles.get_all())


def test_load():
    g = example_grid(-1, 1)
    p = Path("test.db")
    p.unlink(missing_ok=True)
    db_tiles = db.DuckDBTiles.create(g, path=str(p))
    db_tiles.close()

    db_tiles2 = db.DuckDBTiles.load(str(p))
    assert_frame_equal_no_index(g.df, db_tiles2.get_all())


def test_next_tiles():
    g, pd_tiles, db_tiles = prepped_dbs()

    assert_frame_equal_no_index(pd_tiles.next(3, "theta0"), db_tiles.next(3, "theta0"))


def test_bias():
    g = example_grid(-1, 1)

    nB = 11
    data = np.random.rand(g.n_tiles, 1 + nB)
    cols = ["lams"] + [f"B_lams{i}" for i in range(nB)]
    g = g.add_cols(pd.DataFrame(data, index=g.df.index, columns=cols))

    pd_tiles = db.PandasTiles.create(g)
    db_tiles = db.DuckDBTiles.create(g)

    assert pd_tiles.bias() == db_tiles.bias()
