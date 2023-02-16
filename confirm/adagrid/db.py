import contextlib
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd

import confirm.adagrid.json as json
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
    reports: List[Dict] = field(default_factory=list)
    _next_worker_id: int = 2
    _tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    step_info: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    lock = contextlib.suppress()
    is_distributed: bool = False
    supports_threads: bool = False

    @property
    def store(self) -> Store:
        return PandasStore(self._tables)

    def dimension(self) -> int:
        return (
            max([int(c[5:]) for c in self._tiles_columns() if c.startswith("theta")])
            + 1
        )

    def heartbeat(self, worker_id: int):
        return contextlib.AsyncExitStack()

    def _tiles_columns(self) -> List[str]:
        return self.tiles_df.columns

    def _results_columns(self) -> List[str]:
        return self.results_df.columns

    def get_coordination_id(self):
        return 0

    def get_starting_step_id(self, worker_id: int):
        if self.results_df is None:
            return 0
        else:
            return self.results_df["step_id"].max()

    def get_tiles(self) -> pd.DataFrame:
        return self.tiles_df.reset_index(drop=True)

    def get_results(self) -> pd.DataFrame:
        return self.results_df.reset_index(drop=True)

    def insert_report(self, report):
        self.reports.append(report)

    def get_reports(self):
        return pd.DataFrame(self.reports)

    def set_step_info(self, worker_id, step_id, n_tiles, n_packets):
        self.step_info[step_id] = (n_tiles, n_packets)

    def get_step_info(self, worker_id, step_id):
        return self.step_info[step_id]

    def n_processed_tiles(self, worker_id: int, step_id: int) -> int:
        ids = self.tiles_df.loc[self.tiles_df["step_id"] == step_id, "id"]
        return np.in1d(self.results_df["id"], ids).sum()

    def insert_tiles(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        self.tiles_df = pd.concat((self.tiles_df, df), axis=0)

    def insert_results(self, df: pd.DataFrame, orderer: str) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        if self.results_df is None:
            self.results_df = df
        else:
            self.results_df = pd.concat((self.results_df, df), axis=0)

    def get_packet(
        self, coordination_id: int, worker_id: int, step_id: int, packet_id: int
    ) -> pd.DataFrame:
        where = (self.tiles_df["step_id"] == step_id) & (
            self.tiles_df["packet_id"] == packet_id
        )
        return self.tiles_df.loc[where]

    def check_packet_flag(self, worker_id, step_id, packet_id):
        return None

    def set_packet_flag(self, worker_id, step_id, packet_id):
        return True

    def next(
        self, coordination_id: int, worker_id: int, n: int, order_col: str
    ) -> pd.DataFrame:
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

    def bootstrap_lamss(self, worker_id: int) -> pd.Series:
        nB = (
            max([int(c[6:]) for c in self.results_df.columns if c.startswith("B_lams")])
            + 1
        )
        active_tiles = self.results_df.loc[self.results_df["active"]]
        return active_tiles[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0)

    def worst_tile(self, worker_id: int, orderer: str) -> pd.DataFrame:
        active_tiles = self.results_df.loc[self.results_df["active"]]
        return active_tiles.loc[[active_tiles[orderer].idxmin()]]

    async def init_tiles(self, df: pd.DataFrame, wait=False) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        self.tiles_df = df

    def new_workers(self, n) -> List[int]:
        return [self._new_worker() for _ in range(n)]

    def _new_worker(self) -> int:
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
    _tiles_columns_cache: List[str] = None
    _results_columns_cache: List[str] = None
    _d: int = None
    is_distributed: bool = False
    supports_threads: bool = False

    def __post_init__(self):
        self.store = DuckDBStore(self.con)
        self.con.execute(
            """
            create table if not exists packet_flags
                (worker_id int, step_id int, packet_id int)
            """
        )

    def heartbeat(self, worker_id):
        # null-op context manager.
        return contextlib.AsyncExitStack()

    def dimension(self):
        if self._d is None:
            cols = self.con.execute("select * from tiles limit 0").df().columns
            self._d = max([int(c[5:]) for c in cols if c.startswith("theta")]) + 1
        return self._d

    def _tiles_columns(self):
        if self._tiles_columns_cache is None:
            self._tiles_columns_cache = (
                self.con.execute("select * from tiles limit 0").df().columns
            )
        return self._tiles_columns_cache

    def _results_columns(self):
        if self._results_columns_cache is None:
            self._results_columns_cache = (
                self.con.execute("select * from results limit 0").df().columns
            )
        return self._results_columns_cache

    def get_tiles(self):
        return self.con.execute("select * from tiles").df()

    def get_results(self):
        return self.con.execute("select * from results").df()

    def get_reports(self):
        json_strs = self.con.execute("select * from reports").fetchall()
        return pd.DataFrame([json.loads(s[0]) for s in json_strs])

    def set_step_info(self, worker_id, step_id, n_tiles, n_packets):
        pass

    def get_step_info(self, worker_id, step_id):
        n_tiles = self.con.query(
            f"select count(*) from tiles where step_id = {step_id}"
        ).fetchone()[0]
        n_packets = self.con.query(
            f"select count(distinct packet_id) from tiles where step_id = {step_id}"
        ).fetchone()[0]
        return n_tiles, n_packets

    def insert_report(self, report):
        self.con.execute(f"insert into reports values ('{json.dumps(report)}')")

    def n_processed_tiles(self, worker_id: int, step_id: int) -> int:
        # worker_id is ignored because DuckDB only supports one worker.
        return self.con.execute(
            f"""
            select count(*) from tiles
                where
                    step_id = {step_id}
                    and id in (select id from results where step_id = {step_id})
        """
        ).fetchone()[0]

    def get_coordination_id(self):
        return 0

    def get_starting_step_id(self, worker_id):
        if not self._results_table_exists():
            return 0
        return self.con.query(
            "select coalesce(max(step_id), 0) from results"
        ).fetchone()[0]

    def insert_tiles(self, df: pd.DataFrame):
        column_order = ",".join(self._tiles_columns())
        self.con.execute(f"insert into tiles select {column_order} from df")

    def insert_results(self, df: pd.DataFrame, orderer: str):
        if not self._results_table_exists():
            self.con.execute("create table if not exists results as select * from df")
            return
        column_order = ",".join(self._results_columns())
        self.con.execute(f"insert into results select {column_order} from df")

    def _results_table_exists(self):
        out = self.con.query(
            """
            select name from sqlite_master 
                where type='table' 
                and name='results'
            """
        ).fetchall()
        if len(out) == 0:
            return False
        return True

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

    def worst_tile(self, worker_id, order_col):
        # worker_id is ignored because DuckDB only supports one worker.
        return self.con.execute(
            f"select * from results where active=true order by {order_col} limit 1"
        ).df()

    def get_packet(
        self, coordination_id: int, worker_id: int, step_id: int, packet_id: int
    ):
        # coordination_id and worker_id are ignored because DuckDB only
        # supports one worker.
        return self.con.execute(
            f"""
            select * from tiles
                where
                    step_id = {step_id}
                    and packet_id = {packet_id}
            """,
        ).df()

    def check_packet_flag(self, worker_id, step_id, packet_id):
        rows = self.con.query(
            f"""
            select * from packet_flags 
                where worker_id = {worker_id}
                and step_id = {step_id}
                and packet_id = {packet_id}
            """
        ).fetchall()
        if len(rows) == 0:
            return None
        else:
            flag_tuple = rows[0]
            assert flag_tuple[1] == step_id
            assert flag_tuple[2] == packet_id
            return flag_tuple[0]

    def set_packet_flag(self, worker_id, step_id, packet_id):
        if self.check_packet_flag(worker_id, step_id, packet_id) is None:
            self.con.execute(
                f"""
                insert into packet_flags values (
                    {worker_id}, {step_id}, {packet_id}
                )
            """
            )
            return True
        else:
            return False

    def next(
        self, coordination_id: int, worker_id: int, n: int, orderer: str
    ) -> pd.DataFrame:
        # coordination_id and worker_id are ignored because DuckDB only
        # supports one worker.
        #
        # we wrap with a transaction to ensure that concurrent readers don't
        # grab the same chunk of work.
        t = self.con.begin()
        out = t.execute(
            f"""
            select * from results where eligible=true
            order by {orderer} limit {n}
            """
        ).df()
        t.commit()
        return out

    def bootstrap_lamss(self, worker_id: int) -> List[float]:
        # worker_id is ignored because DuckDB only supports one worker.
        # Get the number of bootstrap lambda* columns
        nB = (
            max([int(c[6:]) for c in self._results_columns() if c.startswith("B_lams")])
            + 1
        )

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        lamss = self.con.execute(
            f"select {cols} from results where active=true"
        ).fetchall()[0]

        return lamss

    def close(self) -> None:
        self.con.close()

    def new_workers(self, n: int) -> List[int]:
        return [self._new_worker() for _ in range(n)]

    def _new_worker(self) -> int:
        self.con.execute(
            "create sequence if not exists worker_id start with 1 increment by 1"
        )
        worker_id = self.con.execute("select nextval('worker_id')").fetchone()[0] + 1
        self.con.execute("create table if not exists workers (id int)")
        self.con.execute(f"insert into workers values ({worker_id})")
        return worker_id

    async def init_tiles(self, df: pd.DataFrame, wait: bool = False) -> None:
        self.con.execute("create table tiles as select * from df")
        self.con.execute(
            """
            create table done (
                    coordination_id UINTEGER,
                    worker_id UINTEGER,
                    step_id UINTEGER,
                    packet_id INTEGER,
                    id UBIGINT,
                    active BOOL,
                    finisher_id UINTEGER,
                    refine BOOL,
                    deepen BOOL,
                    split BOOL)
            """
        )
        self.con.execute("insert into done values (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)")
        self.con.execute(
            """
            create table reports (json TEXT)
            """
        )

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
