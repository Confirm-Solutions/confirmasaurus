from pathlib import Path

import numpy as np
import pandas as pd

import imprint as ip
from confirm.adagrid import db


def example_grid(x1, x2):
    N = 10
    theta, radii = ip.grid._cartesian_gridpts([x1], [x2], [N])
    H = ip.planar_null.HyperPlane(np.array([-1]), 0)
    g = ip.create_grid(theta, radii=radii, null_hypos=[H])
    # Typically this field would be set by the adagrid code.
    g.df["step_id"] = 17
    g.df["step_iter"] = np.arange(g.df.shape[0])
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
        g.df.index = g.df.index.astype(np.uint64)
        pd_tiles = db.PandasTiles()
        pd_tiles.init_tiles(g.df)
        db_tiles = self.connect()
        db_tiles.init_tiles(g.df)
        return g, pd_tiles, db_tiles

    def insert_fake_results(self, db):
        work = db.get_tiles().nsmallest(100, "theta0")
        work["orderer"] = np.linspace(5, 6, work.shape[0])
        work["eligible"] = True
        db.insert_results(work, "orderer")

    def test_connect(self):
        self.connect()

    def test_create(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()
        pd.testing.assert_frame_equal(pd_tiles.get_tiles(), g.df.reset_index(drop=True))
        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())

    def test_write_tiles(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        g2 = example_grid(-2, -1)
        pd_tiles.insert_tiles(g2.df)
        db_tiles.insert_tiles(g2.df)

        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())

    def test_write_results(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        self.insert_fake_results(pd_tiles)
        self.insert_fake_results(db_tiles)

        assert_frame_equal_special(pd_tiles.get_results(), db_tiles.get_results())

    def test_write_column_ordering(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        g2 = example_grid(-2, -1)
        cols = g2.df.columns.tolist()
        cols.remove("theta0")
        cols.append("theta0")
        g2.df = g2.df[cols]
        pd_tiles.insert_tiles(g2.df)
        db_tiles.insert_tiles(g2.df)

        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())

    def test_set_get_step_info(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()
        for D in [pd_tiles, db_tiles]:
            D.set_step_info(step_id=0, step_iter=1, n_iter=2, n_tiles=3)
            assert D.get_step_info() == (0, 1, 2, 3)

    def test_worst_tile(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        self.insert_fake_results(pd_tiles)
        self.insert_fake_results(db_tiles)
        np.testing.assert_allclose(
            pd_tiles.worst_tile("orderer").iloc[0]["theta0"], -0.9
        )
        np.testing.assert_allclose(
            db_tiles.worst_tile("orderer").iloc[0]["theta0"], -0.9
        )
        assert_frame_equal_special(
            pd_tiles.worst_tile("orderer"), db_tiles.worst_tile("orderer")
        )

    def test_next(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()
        self.insert_fake_results(pd_tiles)
        self.insert_fake_results(db_tiles)

        pd_work = pd_tiles.next(3, "theta0")
        db_work = db_tiles.next(3, "theta0")
        assert_frame_equal_special(pd_work, db_work)
        assert_frame_equal_special(pd_tiles.get_results(), db_tiles.get_results())

        def finish(db_work, pd_work):
            cols = [
                "id",
                "step_id",
                "step_iter",
                "active",
                "finisher_id",
                "refine",
                "deepen",
                "split",
            ]
            for df in [pd_work, db_work]:
                for i, col in enumerate(cols):
                    if col not in df.columns:
                        df[col] = i
            db_work["active"] = False
            pd_work["active"] = False
            db_tiles.finish(db_work[cols])
            pd_tiles.finish(pd_work[cols])

        finish(db_work, pd_work)
        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())

        pd_work = pd_tiles.next(7, "theta0")
        db_work = db_tiles.next(7, "theta0")
        assert pd_work.shape[0] == 2
        assert db_work.shape[0] == 2
        assert_frame_equal_special(pd_work, db_work)

        finish(db_work, pd_work)
        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())
        assert not pd_tiles.get_results()["eligible"].any()
        assert not db_tiles.get_results()["eligible"].any()
        assert not pd_tiles.get_tiles()["active"].any()
        assert not db_tiles.get_tiles()["active"].any()

    def test_get_work(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        pd_work = pd_tiles.get_work(17, 5)
        db_work = db_tiles.get_work(17, 5)
        assert_frame_equal_special(pd_work, db_work)

        # pd_work = pd_tiles.next(3, "theta0", 0)
        # db_work = db_tiles.next(3, "theta0", 0)
        # assert_frame_equal_special(pd_work, db_work)

        # assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

        # pd_work2 = pd_tiles.next(3, "theta0", 0)
        # db_work2 = db_tiles.next(3, "theta0", 0)
        # assert pd_work2.shape[0] == 2
        # assert_frame_equal_special(pd_work2, db_work2)

        # assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

        # db_work["active"] = False
        # pd_work["active"] = False
        # db_tiles.finish(db_work)
        # pd_tiles.finish(pd_work)
        # assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())

        # db_work2["active"] = False
        # pd_work2["active"] = False
        # db_tiles.finish(db_work2)
        # pd_tiles.finish(pd_work2)
        # assert_frame_equal_special(pd_tiles.get_all(), db_tiles.get_all())
        # assert not pd_tiles.get_all()["active"].any()
        # assert not db_tiles.get_all()["active"].any()

    def test_bootstrap_lamss(self):
        g = example_grid(-1, 1)

        nB = 11
        data = np.random.rand(g.n_tiles, 1 + nB)
        cols = ["lams"] + [f"B_lams{i}" for i in range(nB)]
        g = g.add_cols(pd.DataFrame(data, index=g.df.index, columns=cols))
        g.df["orderer"] = np.linspace(-1, 1, g.n_tiles)

        pd_tiles = db.PandasTiles()
        pd_tiles.init_tiles(g.df)
        pd_tiles.insert_results(g.df, "orderer")
        db_tiles = self.connect()
        db_tiles.init_tiles(g.df)
        db_tiles.insert_results(g.df, "orderer")

        np.testing.assert_allclose(
            pd_tiles.bootstrap_lamss(), db_tiles.bootstrap_lamss()
        )

    def test_new_worker(self):
        db = self.connect()
        assert db.new_worker() == 2
        assert db.new_worker() == 3
        assert db.new_worker() == 4


class TestDuckDB(DBTester):
    def connect(self):
        return db.DuckDBTiles.connect()

    def test_duckdb_load(self):
        g = example_grid(-1, 1)
        p = Path("test.db")
        p.unlink(missing_ok=True)
        db_tiles = db.DuckDBTiles.connect(path=str(p))
        db_tiles.init_tiles(g.df)
        db_tiles.close()

        db_tiles2 = db.DuckDBTiles.connect(path=str(p))
        assert_frame_equal_special(g.df, db_tiles2.get_tiles())
