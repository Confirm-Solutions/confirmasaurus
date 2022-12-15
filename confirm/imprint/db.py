from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List

import duckdb
import pandas as pd

from confirm.imprint.store import DuckDBStore
from confirm.imprint.store import PandasStore


@dataclass
class PandasTiles:
    """
    A tile database built on top of Pandas DataFrames.

    This is not very efficient because every write call will copy the entire
    database. But it's a useful reference implementation for testing and
    demonstration.
    """

    df: pd.DataFrame = None
    worker_id: int = 0
    _tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def store(self):
        return PandasStore(self._tables)

    def dimension(self):
        return max([int(c[5:]) for c in self.columns() if c.startswith("theta")]) + 1

    def columns(self):
        return self.df.columns

    def get_all(self):
        return self.df

    def next(self, n, order_col):
        out = self.df.loc[self.df["eligible"]].nsmallest(n, order_col)
        self.df.loc[out.index, "eligible"] = False
        return out

    def finish(self, df):
        self.df.loc[df.index, "active"] = df["active"]

    def bootstrap_lamss(self):
        nB = max([int(c[6:]) for c in self.df.columns if c.startswith("B_lams")]) + 1
        active_tiles = self.df.loc[self.df["active"]]
        return active_tiles[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0)

    def worst_tile(self, order_col):
        active_tiles = self.df.loc[self.df["active"]]
        return active_tiles.loc[[active_tiles[order_col].idxmin()]]

    def write(self, df):
        self.df = pd.concat((self.df, df), axis=0, ignore_index=True)

    def init_tiles(self, df):
        self.df = df.reset_index(drop=True)


@dataclass
class DuckDBTiles:
    """
    A tile database built on top of DuckDB. This should be very fast and
    robust and is the default database for confirm.

    See this GitHub issue for a discussion of the design:
    https://github.com/Confirm-Solutions/confirmasaurus/issues/95
    """

    con: duckdb.DuckDBPyConnection
    # TODO: despite not supporting multiple workers, it would still be good to
    # distinguish between separate runs with worker_id
    worker_id: int = 0
    _columns: List[str] = None
    _d: int = None
    store: DuckDBStore = None

    def __post_init__(self):
        self.store = DuckDBStore(self.con)

    def dimension(self):
        if self._d is None:
            self._d = (
                max([int(c[5:]) for c in self.columns() if c.startswith("theta")]) + 1
            )
        return self._d

    def columns(self):
        if self._columns is None:
            self._columns = self.con.execute("select * from tiles limit 0").df().columns
        return self._columns

    def get_all(self):
        return self.con.execute("select * from tiles").df()

    def write(self, df):
        column_order = ",".join(self.columns())
        self.con.execute(f"insert into tiles select {column_order} from df")

    def next(self, n, order_col):
        # we wrap with a transaction to ensure that concurrent readers don't
        # grab the same chunk of work.
        t = self.con.begin()
        out = t.execute(
            "select * from tiles where eligible=true"
            f" order by {order_col} asc limit {n}"
        ).df()
        t.execute("update tiles set eligible=false where id in (select id from out)")
        t.commit()
        return out

    def finish(self, which):
        self.con.execute(
            "update tiles set active=w.active from which w" " where tiles.id=w.id"
        )

    def bootstrap_lamss(self):
        # Get the number of bootstrap lambda* columns
        nB = self.con.execute(
            "select max(cast(substring(column_name, 7, 10) as int)) + 1"
            "    from information_schema.columns"
            "    where table_name=='tiles' and column_name like 'B_lams%'"
        ).fetchall()[0][0]

        # Get lambda**_Bi for each bootstrap sample.
        lamss = self.con.execute(
            "select "
            + ",".join([f"min(B_lams{i})" for i in range(nB)])
            + " from tiles where active=true"
        ).fetchall()[0]

        return lamss

    def worst_tile(self, order_col):
        return self.con.execute(
            f"select * from tiles where active=true order by {order_col} asc limit 1"
        ).df()

    def close(self):
        self.con.close()

    def init_tiles(self, df):
        self.con.execute("create table tiles as select * from df")

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
