from pathlib import Path

import numpy as np
import pandas as pd

from confirm.adagrid import db
from imprint import grid


def example_grid(x1, x2):
    N = 10
    theta, radii = grid._cartesian_gridpts([x1], [x2], [N])
    H = grid.HyperPlane(np.array([-1]), 0)
    g = grid.init_grid(theta, radii).add_null_hypos([H]).prune()
    # Typically this field would be set by the adagrid code.
    g.df["eligible"] = True
    return g


def assert_frame_equal_special(pd_df, db_df):
    # it's okay for a database backend to return extra columns
    compare_df = db_df.drop(
        [c for c in db_df.columns if c not in pd_df.columns], axis=1
    )

    pd.testing.assert_frame_equal(
        pd_df.sort_values("theta0").reset_index(drop=True),
        compare_df.sort_values("theta0").reset_index(drop=True),
        # TODO: remove
        check_dtype=False,
        check_like=True,
    )


class DBTester:
    def prepped_dbs(self):
        g = example_grid(-1, 1)
        pd_tiles = db.PandasTiles()
        pd_tiles.init_tiles(g.df)
        db_tiles = self.dbtype.connect()
        db_tiles.init_tiles(g.df)
        return g, pd_tiles, db_tiles

    def test_create(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()
        pd.testing.assert_frame_equal(pd_tiles.get_all(), g.df.reset_index(drop=True))
        assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

    def test_write(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        g2 = example_grid(-2, -1)
        pd_tiles.write(g2.df)
        db_tiles.write(g2.df)

        assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

    def test_write_column_ordering(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        g2 = example_grid(-2, -1)
        cols = g2.df.columns.tolist()
        cols.remove("theta0")
        cols.append("theta0")
        g2.df = g2.df[cols]
        pd_tiles.write(g2.df)
        db_tiles.write(g2.df)

        assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

    def test_next_tiles(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        pd_work = pd_tiles.next(3, "theta0")
        db_work = db_tiles.next(3, "theta0")
        assert_frame_equal_special(pd_work, db_work)

        assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

        pd_work2 = pd_tiles.next(3, "theta0")
        db_work2 = db_tiles.next(3, "theta0")
        assert pd_work2.shape[0] == 2
        assert_frame_equal_special(pd_work2, db_work2)

        assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

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

    def test_worst_tile(self):
        np.random.seed(10)
        g = example_grid(-1, 1)
        g.df["lams"] = np.random.rand(g.df.shape[0])
        g.df.loc[g.df["lams"].idxmin(), "active"] = False
        pd_tiles = db.PandasTiles()
        pd_tiles.init_tiles(g.df)
        db_tiles = self.dbtype.connect()
        db_tiles.init_tiles(g.df)
        np.testing.assert_allclose(pd_tiles.worst_tile("lams").iloc[0]["theta0"], -0.1)
        assert_frame_equal_special(
            pd_tiles.worst_tile("lams"), db_tiles.worst_tile("lams")
        )

    def test_bootstrap_lamss(self):
        g = example_grid(-1, 1)

        nB = 11
        data = np.random.rand(g.n_tiles, 1 + nB)
        cols = ["lams"] + [f"B_lams{i}" for i in range(nB)]
        g = g.add_cols(pd.DataFrame(data, index=g.df.index, columns=cols))

        pd_tiles = db.PandasTiles()
        pd_tiles.init_tiles(g.df)
        db_tiles = self.dbtype.connect()
        db_tiles.init_tiles(g.df)

        np.testing.assert_allclose(
            pd_tiles.bootstrap_lamss(), db_tiles.bootstrap_lamss()
        )


class TestDuckDB(DBTester):
    dbtype = db.DuckDBTiles

    def test_duckdb_load(self):
        g = example_grid(-1, 1)
        p = Path("test.db")
        p.unlink(missing_ok=True)
        db_tiles = db.DuckDBTiles.connect(path=str(p))
        db_tiles.init_tiles(g.df)
        db_tiles.close()

        db_tiles2 = db.DuckDBTiles.connect(path=str(p))
        assert_frame_equal_special(g.df, db_tiles2.get_all())
