import logging
from pathlib import Path

import numpy as np
import pandas as pd

import imprint as ip
from confirm.adagrid.db import DatabaseLogging
from confirm.adagrid.db import DuckDBTiles
from confirm.adagrid.db import PandasTiles
from confirm.adagrid.init import _serialize_null_hypos

logger = logging.getLogger(__name__)


def example_grid(x1, x2):
    N = 10
    theta, radii = ip.grid._cartesian_gridpts([x1], [x2], [N])
    H = ip.planar_null.HyperPlane(np.array([-1]), 0)
    g = ip.create_grid(theta, radii=radii, null_hypos=[H])
    # Typically these fields would be set by the adagrid code.
    g.df["step_id"] = 17
    g.df["packet_id"] = np.arange(g.df.shape[0])
    return g


def assert_frame_equal_special(pd_df, db_df):
    # it's okay for a database backend to return extra columns
    compare_df = db_df.drop(
        [c for c in db_df.columns if c not in pd_df.columns], axis=1
    )

    pd.testing.assert_frame_equal(
        pd_df.sort_values("theta0").reset_index(drop=True),
        compare_df.sort_values("theta0").reset_index(drop=True),
        check_dtype=False,
        check_like=True,
    )


class DBTester:
    def prepped_dbs(self):
        g = example_grid(-1, 1)
        g.df.index = g.df.index.astype(np.uint64)
        pd_tiles = PandasTiles()

        pd_tiles.init_grid(g.df)

        db_tiles = self.connect()
        self.init_grid(db_tiles, g)
        return g, pd_tiles, db_tiles

    def init_grid(self, db, g):
        db.init_grid(
            g.df, _serialize_null_hypos(g.null_hypos), pd.DataFrame([dict(a=1)])
        )

    def insert_fake_results(self, db, zone_id=0):
        work = db.get_tiles().nsmallest(100, "theta0")
        work["orderer"] = np.linspace(5, 6, work.shape[0])
        work["eligible"] = True
        work["zone_id"] = zone_id
        db.insert_results(work, "orderer")

    def test_connect(self):
        self.connect()

    def test_create(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()
        pd.testing.assert_frame_equal(pd_tiles.get_tiles(), g.df.reset_index(drop=True))
        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())

    def test_insert_report(self):
        db = self.connect()
        self.init_grid(db, example_grid(-1, 1))
        R = dict(zone_id=1, step_id=2, packet_id=3, testA="testB")
        db.insert_report(R)
        pd.testing.assert_frame_equal(pd.DataFrame([R]), db.get_reports())

    def test_write_tiles(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        g2 = example_grid(-2, -1)
        pd_tiles.insert_tiles(g2.df)
        db_tiles.insert_tiles(g2.df)

        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())

    def test_get_incomplete_packets(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()
        incomplete = [(17, i) for i in range(5)]
        assert pd_tiles.get_incomplete_packets() == incomplete
        assert db_tiles.get_incomplete_packets() == incomplete

        self.insert_fake_results(pd_tiles)
        self.insert_fake_results(db_tiles)
        assert pd_tiles.get_incomplete_packets() == []
        assert db_tiles.get_incomplete_packets() == []

        g.df["id"] = ip.grid._gen_short_uuids(g.df.shape[0], g.worker_id)
        g.df["step_id"] = 20
        g.df["packet_id"] = 0
        pd_tiles.insert_tiles(g.df)
        db_tiles.insert_tiles(g.df)

        assert pd_tiles.get_incomplete_packets() == [(20, 0)]
        assert db_tiles.get_incomplete_packets() == [(20, 0)]

    def test_get_zone_info(self):
        g, _, db_tiles = self.prepped_dbs()
        assert db_tiles.get_next_step() == 18

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

    def test_worst_tile(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        self.insert_fake_results(pd_tiles)
        self.insert_fake_results(db_tiles)
        np.testing.assert_allclose(
            pd_tiles.worst_tile(20, "orderer").iloc[0]["theta0"], -0.9
        )
        np.testing.assert_allclose(
            db_tiles.worst_tile(20, "orderer").iloc[0]["theta0"], -0.9
        )
        assert_frame_equal_special(
            pd_tiles.worst_tile(20, "orderer"), db_tiles.worst_tile(20, "orderer")
        )

    def test_next(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()
        self.insert_fake_results(pd_tiles)
        self.insert_fake_results(db_tiles)

        pd_work = pd_tiles.next(17, 18, 3, "theta0")
        db_work = db_tiles.next(17, 18, 3, "theta0")
        assert_frame_equal_special(pd_work, db_work)
        assert_frame_equal_special(pd_tiles.get_results(), db_tiles.get_results())

        def insert_done(db_work, pd_work):
            cols = [
                "step_id",
                "packet_id",
                "id",
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
            db_tiles.insert_done(db_work[cols])
            pd_tiles.insert_done(pd_work[cols])

        insert_done(db_work, pd_work)
        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())

        pd_work = pd_tiles.next(17, 18, 7, "theta0")
        db_work = db_tiles.next(17, 18, 7, "theta0")
        assert pd_work.shape[0] == 2
        assert db_work.shape[0] == 2
        assert_frame_equal_special(pd_work, db_work)

        insert_done(db_work, pd_work)
        assert_frame_equal_special(pd_tiles.get_tiles(), db_tiles.get_tiles())
        assert not pd_tiles.get_results()["eligible"].any()
        assert not db_tiles.get_results()["eligible"].any()
        assert not pd_tiles.get_tiles()["active"].any()
        assert not db_tiles.get_tiles()["active"].any()

    def test_get_packet(self):
        g, pd_tiles, db_tiles = self.prepped_dbs()

        pd_work = pd_tiles.get_packet(17, 5)
        db_work = db_tiles.get_packet(17, 5)
        assert_frame_equal_special(pd_work, db_work)

    def test_bootstrap_lamss(self):
        g = example_grid(-1, 1)

        nB = 11
        data = np.random.rand(g.n_tiles, 1 + nB)
        cols = ["lams"] + [f"B_lams{i}" for i in range(nB)]
        g = g.add_cols(pd.DataFrame(data, index=g.df.index, columns=cols))
        g.df["orderer"] = np.linspace(-1, 1, g.n_tiles)

        pd_tiles = PandasTiles()
        pd_tiles.init_grid(g.df)
        pd_tiles.insert_results(g.df, "orderer")
        db_tiles = self.connect()
        self.init_grid(db_tiles, g)
        db_tiles.insert_results(g.df, "orderer")

        np.testing.assert_allclose(
            pd_tiles.bootstrap_lamss(17), db_tiles.bootstrap_lamss(17)
        )

    def test_db_logging(self):
        db = self.connect()
        self.init_grid(db, example_grid(-1, 1))
        with DatabaseLogging(db=db):
            logger.debug("informative")
            logger.warning("scary")
            logger.error("ameoba")
        logs = db.get_logs().sort_values(by=["t"])
        assert (logs["name"] == "tests.test_db").all()
        assert logs.iloc[1]["message"] == "scary"
        assert (logs["pathname"] == __file__).all()
        assert logs.shape[0] == 3


class TestDuckDB(DBTester):
    def connect(self, **kwargs):
        return DuckDBTiles.connect()

    def test_duckdb_load(self):
        g = example_grid(-1, 1)
        p = Path("test.db")
        p.unlink(missing_ok=True)
        db_tiles = DuckDBTiles.connect(job_name="test")
        self.init_grid(db_tiles, g)
        db_tiles.close()

        db_tiles2 = DuckDBTiles.connect(job_name="test")
        assert_frame_equal_special(g.df, db_tiles2.get_tiles())
