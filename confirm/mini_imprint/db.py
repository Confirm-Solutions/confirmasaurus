from dataclasses import dataclass
from typing import List

import duckdb
import pandas as pd


@dataclass
class PandasTiles:
    df: pd.DataFrame

    def get_all(self):
        return self.df

    def next(self, n, order_col):
        out = self.df.loc[(~self.df["locked"]) & (self.df["eligible"])].nsmallest(
            n, order_col
        )
        self.df.loc[out.index, "locked"] = True
        self.df.loc[out.index, "eligible"] = False
        return out

    def finish(self, df):
        self.df.loc[df.index, "active"] = df["active"]
        self.df.loc[df.index, "locked"] = False

    def unlock_all(self):
        self.df["locked"] = False

    def bootstrap_lamss(self):
        nB = max([int(c[6:]) for c in self.df.columns if c.startswith("B_lams")]) + 1
        active_tiles = self.df.loc[self.df["active"]]
        return active_tiles[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0)

    def worst_tile(self, order_col):
        active_tiles = self.df.loc[self.df["active"]]
        return active_tiles.loc[[active_tiles[order_col].idxmin()]]

    def write(self, df):
        self.df = pd.concat((self.df, df), axis=0, ignore_index=True)

    @staticmethod
    def create(df):
        out = PandasTiles(df.reset_index(drop=True))
        return out


@dataclass
class DuckDBTiles:
    con: duckdb.DuckDBPyConnection
    columns: List[str]
    column_order: str

    def __init__(self, con):
        self.con = con
        self.columns = self.con.execute("select * from tiles limit 0").df().columns
        self.column_order = ",".join(self.columns)

    def get_all(self):
        return self.con.execute("select * from tiles").df()

    def write(self, df):
        self.con.execute(f"insert into tiles select {self.column_order} from df")

    def next(self, n, order_col):
        # we wrap with a transaction to ensure that concurrent readers don't
        # grab the same chunk of work.
        t = self.con.begin()
        out = t.execute(
            "select * from tiles where locked=false and eligible=true"
            f" order by {order_col} asc limit {n}"
        ).df()
        t.execute(
            "update tiles set locked=true, eligible=false where id in"
            " (select id from out)"
        )
        t.commit()
        return out

    def finish(self, which):
        self.con.execute(
            "update tiles set locked=false, active=w.active from which w"
            " where tiles.id=w.id"
        )

    def unlock_all(self):
        self.con.execute("update tiles set locked=false")

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

    @staticmethod
    def create(df, path=":memory:"):
        """
        Creating a database from a grid.

        We include the grid in the connection method so that we can create the
        tiles table with the same set of columns as the grid dataframe

        Args:
            g: _description_
            path: _description_. Defaults to ":memory:".

        Returns:
            _description_
        """
        con = duckdb.connect(path)
        con.execute("create table tiles as select * from df")
        return DuckDBTiles(con)

    @staticmethod
    def load(path):
        con = duckdb.connect(path)
        return DuckDBTiles(con)
