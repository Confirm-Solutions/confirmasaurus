from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List

import duckdb
import pandas as pd

from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.store import DuckDBStore
from confirm.adagrid.store import PandasStore


def serial_next(db, convergence_f, n_steps, _, packet_size, order_col, worker_id):
    step_id_df = db.store.get("step_id")
    if step_id_df["step_id"].iloc[0] >= n_steps - 1:
        return WorkerStatus.REACHED_N_STEPS, None, dict()

    step_id_df["step_id"] += 1
    work = db.get_work(packet_size, order_col, worker_id)
    step_id_df = db.store.set("step_id", step_id_df)

    if convergence_f():
        status = WorkerStatus.CONVERGED
    elif work.shape[0] == 0:
        status = WorkerStatus.FAILED
    else:
        status = WorkerStatus.WORKING

    return status, work, dict()


@dataclass
class PandasTiles:
    """
    A tile database built on top of Pandas DataFrames.

    This is not very efficient because every write call will copy the entire
    database. But it's a useful reference implementation for testing and
    demonstration.
    """

    df: pd.DataFrame = None
    _next_worker_id: int = 2
    _tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    next = serial_next

    @property
    def store(self):
        return PandasStore(self._tables)

    def dimension(self):
        return max([int(c[5:]) for c in self.columns() if c.startswith("theta")]) + 1

    def columns(self):
        return self.df.columns

    def get_all(self):
        return self.df

    def write(self, df):
        self.df = pd.concat((self.df, df), axis=0, ignore_index=True)

    def get_work(self, packet_size, order_col, _):
        out = self.df.loc[self.df["eligible"]].nsmallest(packet_size, order_col)
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

    def init_tiles(self, df):
        self.df = df.reset_index(drop=True)

    def new_worker(self):
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
    _columns: List[str] = None
    _d: int = None
    store: DuckDBStore = None
    next = serial_next

    def __post_init__(self):
        self.store = DuckDBStore(self.con)
        if not self.store.exists("step_id"):
            self.store.set("step_id", pd.DataFrame(dict(step_id=[-1])))

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

    def get_work(self, n, order_col, _):
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
