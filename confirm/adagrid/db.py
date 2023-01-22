import contextlib
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd

import imprint.log
from confirm.adagrid.store import DuckDBStore
from confirm.adagrid.store import PandasStore
from confirm.adagrid.store import Store

logger = imprint.log.getLogger(__name__)


@dataclass
class PandasTiles:
    """
    A tile database built on top of Pandas DataFrames.

    This is not very efficient because every write call will copy the entire
    database. But it's a useful reference implementation for testing and
    demonstration.
    """

    tiles_df: pd.DataFrame = None
    results_df: pd.DataFrame = None
    done_df: pd.DataFrame = None
    _next_worker_id: int = 2
    _tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    step_info = None
    lock = contextlib.suppress()

    @property
    def store(self) -> Store:
        return PandasStore(self._tables)

    def dimension(self) -> int:
        return (
            max([int(c[5:]) for c in self.tiles_columns() if c.startswith("theta")]) + 1
        )

    def tiles_columns(self) -> List[str]:
        return self.tiles_df.columns

    def results_columns(self) -> List[str]:
        return self.results_df.columns

    def get_tiles(self) -> pd.DataFrame:
        return self.tiles_df.reset_index(drop=True)

    def get_results(self) -> pd.DataFrame:
        return self.results_df.reset_index(drop=True)

    def get_step_info(self) -> Tuple[int, int, int, int]:
        return self.step_info

    def set_step_info(
        self, *, step_id: int, step_iter: int, n_iter: int, n_tiles: int
    ) -> None:
        self.step_info = (step_id, step_iter, n_iter, n_tiles)

    def n_processed_tiles(self, step_id: int) -> int:
        ids = self.tiles_df.loc[self.tiles_df["step_id"] == step_id, "id"]
        return np.in1d(self.results_df["id"], ids).sum()

    def insert_tiles(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        self.tiles_df = pd.concat((self.tiles_df, df), axis=0)

    def insert_results(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        if self.results_df is None:
            self.results_df = df
        else:
            self.results_df = pd.concat((self.results_df, df), axis=0)

    def get_work(self, step_id: int, step_iter: int) -> pd.DataFrame:
        where = (self.tiles_df["step_id"] == step_id) & (
            self.tiles_df["step_iter"] == step_iter
        )
        return self.tiles_df.loc[where]

    def select_tiles(self, n: int, order_col: str) -> pd.DataFrame:
        out = self.results_df.loc[self.results_df["eligible"]].nsmallest(n, order_col)
        return out

    def finish(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        if self.done_df is None:
            self.done_df = df
        else:
            self.done_df = pd.concat((self.done_df, df), axis=0)
        self.tiles_df.loc[df["id"], "active"] = df["active"]
        self.results_df.loc[df["id"], "eligible"] = False
        self.results_df.loc[df["id"], "active"] = df["active"]

    def bootstrap_lamss(self) -> pd.Series:
        nB = (
            max([int(c[6:]) for c in self.results_df.columns if c.startswith("B_lams")])
            + 1
        )
        active_tiles = self.results_df.loc[self.results_df["active"]]
        return active_tiles[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0)

    def worst_tile(self, order_col: str) -> pd.DataFrame:
        active_tiles = self.results_df.loc[self.results_df["active"]]
        return active_tiles.loc[[active_tiles[order_col].idxmin()]]

    def init_tiles(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        self.tiles_df = df

    def new_worker(self) -> int:
        out = self._next_worker_id
        self._next_worker_id += 1
        return out


@dataclass
class DuckDBTiles:
    """
    A tile database built on top of DuckDB. This should be very fast and
    robust and is the default database for confirm.

    See this GitHub issue for a discussion of the design:
    https://github.com/Confirm-Solutions/confirmasaurus/issues/95
    """

    con: duckdb.DuckDBPyConnection
    store: DuckDBStore = None
    lock = contextlib.suppress()
    _tiles_columns: List[str] = None
    _results_columns: List[str] = None
    _d: int = None
    _results_table_exists: bool = False

    def __post_init__(self):
        self.store = DuckDBStore(self.con)
        if not self.store.exists("step_id"):
            self.store.set("step_id", pd.DataFrame(dict(step_id=[-1])))

    def dimension(self):
        if self._d is None:
            cols = self.con.execute("select * from tiles limit 0").df().columns
            self._d = max([int(c[5:]) for c in cols if c.startswith("theta")]) + 1
        return self._d

    def tiles_columns(self):
        if self._tiles_columns is None:
            self._tiles_columns = (
                self.con.execute("select * from tiles limit 0").df().columns
            )
        return self._tiles_columns

    def results_columns(self):
        if self._results_columns is None:
            self._results_columns = (
                self.con.execute("select * from results limit 0").df().columns
            )
        return self._results_columns

    def get_tiles(self):
        return self.con.execute("select * from tiles").df()

    def get_results(self):
        return self.con.execute("select * from results").df()

    def get_step_info(self):
        s = self.con.execute("select * from step_info").df().iloc[0]
        return s["step_id"], s["step_iter"], s["n_iter"], s["n_tiles"]

    def set_step_info(self, *, step_id, step_iter, n_iter, n_tiles):
        self.con.execute("delete from step_info")
        self.con.execute(
            "insert into step_info values "
            f"({step_id}, {step_iter}, {n_iter}, {n_tiles})"
        )

    def n_processed_tiles(self, step_id):
        return self.con.execute(
            f"""
            select count(*) from tiles
                where
                    step_id = {step_id}
                    and id in (select id from results where step_id = {step_id})
        """
        ).fetchone()[0]

    def insert_tiles(self, df):
        column_order = ",".join(self.tiles_columns())
        self.con.execute(f"insert into tiles select {column_order} from df")

    def insert_results(self, df):
        if not self._results_table_exists:
            self.con.execute("create table if not exists results as select * from df")
            self._results_table_exists = True
            return
        column_order = ",".join(self.results_columns())
        self.con.execute(f"insert into results select {column_order} from df")

    def worst_tile(self, order_col):
        return self.con.execute(
            f"select * from results where active=true order by {order_col} asc limit 1"
        ).df()

    def get_work(self, step_id, step_iter):
        return self.con.execute(
            f"""
            select * from tiles
                where
                    step_id = {step_id}
                    and step_iter = {step_iter}
            """,
        ).df()

    def select_tiles(self, n, order_col):
        # we wrap with a transaction to ensure that concurrent readers don't
        # grab the same chunk of work.
        t = self.con.begin()
        out = t.execute(
            f"""
            select * from results where eligible=true
            order by {order_col} asc limit {n}
            """,
        ).df()
        t.commit()
        return out

    def finish(self, which):
        logger.debug(f"finish: {which.head()}")
        column_order = ",".join(which.columns)
        self.con.execute(f"insert into done select {column_order} from which")
        self.con.execute(
            "update tiles set active=w.active from which w where tiles.id=w.id"
        )
        self.con.execute(
            """
            update results
                set eligible=false, active=w.active
            from which w where results.id=w.id
            """
        )

    def bootstrap_lamss(self):
        # Get the number of bootstrap lambda* columns
        # Get the number of bootstrap lambda* columns
        nB = (
            max([int(c[6:]) for c in self.results_columns() if c.startswith("B_lams")])
            + 1
        )

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        lamss = self.con.execute(
            f"select {cols} from results where active=true"
        ).fetchall()[0]

        return lamss

    def close(self):
        self.con.close()

    def init_tiles(self, df):
        self.con.execute("create table tiles as select * from df")
        self.con.execute(
            """
            create table done (
                    id UBIGINT,
                    step_id UINTEGER,
                    step_iter UINTEGER,
                    active BOOL,
                    query_time DOUBLE,
                    finisher_id UINTEGER,
                    refine BOOL,
                    deepen BOOL)
            """
        )
        self.con.execute(
            """
            create table step_info (
                    step_id UINTEGER,
                    step_iter UINTEGER,
                    n_iter UINTEGER,
                    n_tiles UBIGINT)
            """
        )
        self.con.execute("insert into done values (0, 0, 0, 0, 0, 0, 0, 0)")

    def new_worker(self):
        self.con.execute(
            "create sequence if not exists worker_id start with 1 increment by 1"
        )
        worker_id = self.con.execute("select nextval('worker_id')").fetchone()[0] + 1
        self.con.execute("create table if not exists workers (id int)")
        self.con.execute(f"insert into workers values ({worker_id})")
        return worker_id

    @staticmethod
    def connect(path=":memory:"):
        """
        Load a tile database from a file.

        Args:
            path: The filepath to the database.

        Returns:
            The tile database.
        """
        return DuckDBTiles(duckdb.connect(path))
