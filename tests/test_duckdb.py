from pathlib import Path

import numpy as np
import pandas as pd

from confirm.mini_imprint import db
from confirm.mini_imprint import grid


def example_grid(x1, x2):
    N = 10
    theta, radii = grid.cartesian_gridpts([x1], [x2], [N])
    return grid.init_grid(theta, radii, 50).add_null_hypo(0).prune()


def assert_frame_equal_special(pd_df, db_df):
    pd.testing.assert_frame_equal(pd_df.reset_index(drop=True), db_df)


def prepped_dbs():
    g = example_grid(-1, 1)
    pd_tiles = db.PandasTiles.create(g.df)
    db_tiles = db.DuckDBTiles.create(g.df)
    return g, pd_tiles, db_tiles


def test_create():
    g, pd_tiles, db_tiles = prepped_dbs()
    pd.testing.assert_frame_equal(
        pd_tiles.get_all().drop("id", axis=1), g.df.reset_index(drop=True)
    )
    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())


def test_write():
    g, pd_tiles, db_tiles = prepped_dbs()

    g2 = example_grid(-2, -1)
    pd_tiles.write(g2.df)
    db_tiles.write(g2.df)

    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())


def test_load():
    g = example_grid(-1, 1)
    p = Path("test.db")
    p.unlink(missing_ok=True)
    db_tiles = db.DuckDBTiles.create(g.df, path=str(p))
    db_tiles.close()

    db_tiles2 = db.DuckDBTiles.load(str(p))
    assert_frame_equal_special(
        g.df.reset_index(drop=True), db_tiles2.get_all().drop("id", axis=1)
    )


def test_next_tiles():
    g, pd_tiles, db_tiles = prepped_dbs()

    pd_work = pd_tiles.next(3, "theta0")
    db_work = db_tiles.next(3, "theta0")
    assert_frame_equal_special(pd_work, db_work)

    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

    pd_work2 = pd_tiles.next(3, "theta0")
    db_work2 = db_tiles.next(3, "theta0")
    assert pd_work2.shape[0] == 2
    assert_frame_equal_special(pd_work2, db_work2)

    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

    assert db_tiles.get_all()["locked"].all()

    db_tiles.finish(db_work)
    pd_tiles.finish(pd_work, pd_work["K"] > 0)
    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

    db_tiles.finish(db_work2)
    pd_tiles.finish(pd_work2, pd_work2["K"] > 0)
    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())
    assert not pd_tiles.get_all()["locked"].any()
    assert not db_tiles.get_all()["locked"].any()


def test_worst_tile():
    g = example_grid(-1, 1)
    g.df["lams"] = np.random.rand(g.df.shape[0])
    g.df.loc[g.df["lams"].idxmin(), "active"] = False
    pd_tiles = db.PandasTiles.create(g.df)
    db_tiles = db.DuckDBTiles.create(g.df)
    assert_frame_equal_special(pd_tiles.worst_tile(), db_tiles.worst_tile())


def test_bootstrap_lamss():
    g = example_grid(-1, 1)

    nB = 11
    data = np.random.rand(g.n_tiles, 1 + nB)
    cols = ["lams"] + [f"B_lams{i}" for i in range(nB)]
    g = g.add_cols(pd.DataFrame(data, index=g.df.index, columns=cols))

    pd_tiles = db.PandasTiles.create(g.df)
    db_tiles = db.DuckDBTiles.create(g.df)

    np.testing.assert_allclose(pd_tiles.bootstrap_lamss(), db_tiles.bootstrap_lamss())
