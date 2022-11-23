from pathlib import Path

import numpy as np
import pandas as pd

from confirm.imprint import db
from confirm.imprint import grid


def example_grid(x1, x2):
    N = 10
    theta, radii = grid._cartesian_gridpts([x1], [x2], [N])
    H = grid.HyperPlane(np.array([-1]), 0)
    return grid.init_grid(theta, radii).add_null_hypos([H]).prune()


def assert_frame_equal_special(pd_df, db_df):
    pd.testing.assert_frame_equal(pd_df.reset_index(drop=True), db_df)


def prepped_dbs():
    g = example_grid(-1, 1)
    pd_tiles = db.PandasTiles.create(g.df)
    db_tiles = db.DuckDBTiles.create(g.df)
    return g, pd_tiles, db_tiles


def test_create():
    g, pd_tiles, db_tiles = prepped_dbs()
    pd.testing.assert_frame_equal(pd_tiles.get_all(), g.df.reset_index(drop=True))
    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())


def test_write():
    g, pd_tiles, db_tiles = prepped_dbs()

    g2 = example_grid(-2, -1)
    pd_tiles.write(g2.df)
    db_tiles.write(g2.df)

    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())


def test_write_column_ordering():
    g, pd_tiles, db_tiles = prepped_dbs()

    g2 = example_grid(-2, -1)
    cols = g2.df.columns.tolist()
    cols.remove("theta0")
    cols.append("theta0")
    g2.df = g2.df[cols]
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
    assert_frame_equal_special(g.df, db_tiles2.get_all())


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

    assert (~db_tiles.get_all()["eligible"]).all()

    db_work["active"] = False
    pd_work["active"] = False
    db_tiles.finish(db_work)
    pd_tiles.finish(pd_work)
    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

    db_work2["active"] = False
    pd_work2["active"] = False
    db_tiles.finish(db_work2)
    pd_tiles.finish(pd_work2)
    assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())
    assert not pd_tiles.get_all()["active"].any()
    assert not db_tiles.get_all()["active"].any()


def test_worst_tile():
    np.random.seed(10)
    g = example_grid(-1, 1)
    g.df["lams"] = np.random.rand(g.df.shape[0])
    g.df.loc[g.df["lams"].idxmin(), "active"] = False
    pd_tiles = db.PandasTiles.create(g.df)
    db_tiles = db.DuckDBTiles.create(g.df)
    np.testing.assert_allclose(pd_tiles.worst_tile("lams").iloc[0]["theta0"], -0.1)
    assert_frame_equal_special(pd_tiles.worst_tile("lams"), db_tiles.worst_tile("lams"))


def test_bootstrap_lamss():
    g = example_grid(-1, 1)

    nB = 11
    data = np.random.rand(g.n_tiles, 1 + nB)
    cols = ["lams"] + [f"B_lams{i}" for i in range(nB)]
    g = g.add_cols(pd.DataFrame(data, index=g.df.index, columns=cols))

    pd_tiles = db.PandasTiles.create(g.df)
    db_tiles = db.DuckDBTiles.create(g.df)

    np.testing.assert_allclose(pd_tiles.bootstrap_lamss(), db_tiles.bootstrap_lamss())
