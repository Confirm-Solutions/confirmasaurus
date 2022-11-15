from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd


@dataclass
class PandasTiles:
    df: pd.DataFrame

    def get_all(self):
        return self.df

    def next(self, n, order_col):
        return self.df.nlargest(n, order_col)

    def bias(self):
        nB = max([int(c[6:]) for c in self.df.columns if c.startswith("B_lams")]) + 1
        return (
            self.df["lams"].min(axis=0)
            - self.df[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0).mean()
        )

    def write(self, g):
        self.df = pd.concat((self.df, g.df), axis=0)

    @staticmethod
    def create(g):
        return PandasTiles(g.df)


@dataclass
class DuckDBTiles:
    con: duckdb.DuckDBPyConnection

    def get_all(self):
        return self.con.execute("select * from tiles").df()

    def write(self, g):
        tile_df = g.df  # noqa
        self.con.execute("insert into tiles select * from tile_df")

    def next(self, n, order_col):
        return self.con.execute(
            f"select * from tiles order by {order_col} limit {n}"
        ).df()

    def bias(self):
        # Get the number of bootstrap lambda* columns
        nB = self.con.execute(
            "select max(cast(substring(column_name, 7, 10) as int)) + 1"
            "    from information_schema.columns"
            "    where table_name=='tiles' and column_name like 'B_lams%'"
        ).fetchall()[0][0]

        # Get lambda**_Bi for each bootstrap sample.
        lamss = self.con.execute(
            "select min(lams),"
            + ",".join([f"min(B_lams{i})" for i in range(nB)])
            + " from tiles"
        ).fetchall()[0]

        # Calculate the bias in lambda**.
        return lamss[0] - np.mean(lamss[1:])

    def close(self):
        self.con.close()

    @staticmethod
    def create(g, path=":memory:"):
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
        tile_df = g.df  # noqa
        con.execute("create table tiles as select *  from tile_df")
        return DuckDBTiles(con)

    @staticmethod
    def load(path):
        con = duckdb.connect(path)
        return DuckDBTiles(con)
