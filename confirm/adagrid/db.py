import logging.handlers
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

import pandas as pd

import confirm.adagrid.json as json
from .const import MAX_STEP

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)


class DatabaseLogging(logging.handlers.BufferingHandler):
    """
    A logging handler context manager that buffers log record writes to a
    database.
    - whenever `capacity` log records are buffered, they flushed to the database.
    - whenever a log is `flushLevel` or higher, all buffered log records are flushed.
    - whenever more than `interval` seconds have elapsed since the last flush,
      all buffered log records are flushed.
    - when the context manager exits, any remaining log records are flushed.

    Check out the documentation and source for logging.Handler and
    logging.handlers.BufferingHandler to understand more about what's going on
    here.
    """

    def __init__(self, db, capacity=100, interval=15, flushLevel=logging.WARNING):
        self.db = db
        self.capacity = capacity
        self.flushLevel = flushLevel
        self.interval = interval
        self.lastFlush = time.time()
        self.creating_thread = threading.current_thread().ident
        self.running = False
        super().__init__(self.capacity)

    def __enter__(self):
        self.running = True
        self.lastFlush = time.time()
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, *_):
        self.close()  # flushes any remaining records
        self.running = False
        logging.getLogger().removeHandler(self)

    def shouldFlush(self, record):
        """
        Check for buffer full or a record at the flushLevel or higher.
        Cribbed from logging.MemoryHandler.
        """
        return (
            (len(self.buffer) >= self.capacity)
            or (record.levelno >= self.flushLevel)
            or (time.time() > self.lastFlush + self.interval)
        ) and (threading.current_thread().ident == self.creating_thread)

    def flush(self):
        """
        Overriden.
        """
        if not self.running:
            return

        self.acquire()
        try:
            if len(self.buffer) == 0:
                return
            df = pd.DataFrame(
                [
                    # See here for a list of record attributes:
                    # https://docs.python.org/3/library/logging.html#logrecord-attributes
                    dict(
                        t=datetime.fromtimestamp(record.created).strftime(
                            "%Y-%m-%d %H:%M:%S.%f"
                        ),
                        name=record.name,
                        pathname=record.pathname,
                        lineno=record.lineno,
                        levelno=record.levelno,
                        levelname=record.levelname,
                        message=self.format(record),
                    )
                    for record in self.buffer
                ]
            )
            self.lastFlush = time.time()
            self.buffer = []
            self.db.insert("logs", df)
        finally:
            self.release()


@dataclass
class PandasTiles:
    """
    A tile database built on top of Pandas DataFrames.

    This is not very efficient because every write call will copy the entire
    database. But it's a useful reference implementation for testing and
    demonstration.
    """

    tiles: pd.DataFrame = None
    results: pd.DataFrame = None
    done: pd.DataFrame = None
    reports: List[Dict] = field(default_factory=list)
    _tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def dimension(self) -> int:
        return (
            max([int(c[5:]) for c in self._tiles_columns() if c.startswith("theta")])
            + 1
        )

    def _tiles_columns(self) -> List[str]:
        return self.tiles.columns

    def _results_columns(self) -> List[str]:
        return self.results.columns

    def does_table_exist(self, table_name: str) -> bool:
        if table_name in ["tiles", "results", "done", "reports"]:
            return getattr(self, table_name) is not None
        return table_name in self._tables

    def get_tiles(self) -> pd.DataFrame:
        return self.tiles.reset_index(drop=True)

    def get_results(self) -> pd.DataFrame:
        return self.results.reset_index(drop=True)

    def get_done(self):
        return self.done.reset_index(drop=True)

    def get_reports(self):
        return pd.DataFrame(self.reports)

    def n_existing_packets(self, step_id):
        return self.tiles[self.tiles["step_id"] == step_id]["packet_id"].max() + 1

    def insert_reports(self, report):
        self.reports.append(report)
        return report

    def insert_tiles(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        self.tiles = pd.concat((self.tiles, df), axis=0)

    def insert_results(self, df: pd.DataFrame, orderer: str) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        if self.results is None:
            self.results = df
        else:
            self.results = pd.concat((self.results, df), axis=0)

    def insert_done(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        if self.done is None:
            self.done = df
        else:
            self.done = pd.concat((self.done, df), axis=0)

        df_inactive = df[df["active"] is False]
        self.tiles.loc[df_inactive["id"], "inactivation_step"] = df_inactive["step_id"]
        self.results.loc[df_inactive["id"], "inactivation_step"] = df_inactive[
            "step_id"
        ]

        df_complete = df[df["active"] is True]
        self.results.loc[df_complete["id"], "completion_step"] = df_complete["step_id"]

    def next(
        self, basal_step_id: int, new_step_id: int, n: int, order_col: str
    ) -> pd.DataFrame:
        out = self.results.loc[
            (self.results["step_id"] <= basal_step_id)
            & (self.results["completion_step"] == MAX_STEP)
        ].nsmallest(n, order_col)
        return out

    def bootstrap_lamss(self, basal_step_id: int) -> pd.Series:
        nB = (
            max([int(c[6:]) for c in self.results.columns if c.startswith("B_lams")])
            + 1
        )
        active_tiles = self.results.loc[
            (self.results["step_id"] <= basal_step_id)
            & (self.results["inactivation_step"] == MAX_STEP)
        ]
        return active_tiles[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0)

    def worst_tile(self, basal_step_id: int, orderer: str) -> pd.DataFrame:
        active_tiles = self.results.loc[
            (self.results["step_id"] <= basal_step_id)
            & (self.results["inactivation_step"] == MAX_STEP)
        ]
        return active_tiles.loc[[active_tiles[orderer].idxmin()]]

    def verify(self):
        pass

    def init_grid(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        self.tiles = df


@dataclass
class SQLTiles:
    def dimension(self):
        return (
            max(
                [int(c[5:]) for c in self.get_columns("tiles") if c.startswith("theta")]
            )
            + 1
        )

    def get_table(self, table_name: str) -> pd.DataFrame:
        return self.query(f"select * from {table_name}")

    def get_results(self) -> pd.DataFrame:
        out = self.get_table("results")
        out.insert(0, "active", out["inactivation_step"] == MAX_STEP)
        return out

    def get_reports(self) -> pd.DataFrame:
        json_strs = self.get_table("reports").values[:, 0]
        return pd.DataFrame([json.loads(s[0]) for s in json_strs])

    def n_existing_packets(self, step_id: int) -> int:
        return self.query(
            f"""
            select max(packet_id) + 1 from tiles
                where step_id = {step_id}
            """
        ).iloc[0][0]

    def insert_reports(self, *reports: Dict[str, Any]) -> None:
        df = pd.DataFrame(dict(json=[json.dumps(R) for R in reports]))
        self.insert("reports", df)

    def next(
        self, basal_step_id: int, new_step_id: int, n: int, orderer: str
    ) -> pd.DataFrame:
        return self.query(
            f"""
            select * from results 
                where completion_step >= {new_step_id}
                    and step_id <= {basal_step_id}
            order by {orderer} limit {n}
            """
        )

    def bootstrap_lamss(self, basal_step_id: int, nB: int) -> List[float]:
        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        return (
            self.query(
                f"""
            select {cols} from results 
                where inactivation_step > {basal_step_id}
                    and step_id <= {basal_step_id}
            """
            )
            .iloc[0]
            .values
        )

    def worst_tile(self, basal_step_id: int, order_col: str) -> pd.DataFrame:
        return self.query(
            f"""
            select * from results
                where inactivation_step > {basal_step_id}
                    and step_id <= {basal_step_id}
                order by {order_col} limit 1
            """
        )

    async def wait_for_step(
        self, basal_step_id: int, step_id: int, expected_counts: Dict[int, int]
    ):
        pass

    def verify(self, step_id: int):
        duplicate_tiles = self.query(
            f"""
            select id from tiles 
                where step_id <= {step_id}
                group by id 
                having count(*) > 1
            """,
            quiet=True,
        )
        if len(duplicate_tiles) > 0:
            raise ValueError(f"Duplicate tiles: {duplicate_tiles}")

        duplicate_results = self.query(
            f"""
            select id from results 
                where step_id <= {step_id} 
                group by id 
                having count(*) > 1
            """,
            quiet=True,
        )
        if len(duplicate_results) > 0:
            raise ValueError(f"Duplicate results: {duplicate_results}")

        duplicate_done = self.query(
            f"""
            select id from done 
                where step_id <= {step_id} 
                group by id 
                having count(*) > 1
            """,
            quiet=True,
        )
        if len(duplicate_done) > 0:
            raise ValueError(f"Duplicate done: {duplicate_done}")

        results_without_tiles = self.query(
            f"""
            select id from results
                where step_id <= {step_id}
                    and id not in (select id from tiles)
            """,
            quiet=True,
        )
        if len(results_without_tiles) > 0:
            raise ValueError(
                "Rows in results without corresponding rows in tiles:"
                f" {results_without_tiles}"
            )

        active_tiles_without_results = self.query(
            f"""
            select id from tiles
                where 
                    step_id <= {step_id}
                    and active_at_birth = true
                    and id not in (select id from results)
            """,
            quiet=True,
        )
        if len(active_tiles_without_results) > 0:
            raise ValueError(
                "Rows in tiles without corresponding rows in results:"
                f" {active_tiles_without_results}"
            )

        tiles_without_parents = self.query(
            f"""
            select parent_id, id from tiles
                where active_at_birth = true
                    and step_id <= {step_id}
                    and parent_id not in (select id from done)
            """,
            quiet=True,
        )
        if len(tiles_without_parents) > 0:
            raise ValueError(f"tiles without parents: {tiles_without_parents}")

        tiles_with_active_or_incomplete_parents = self.query(
            f"""
            select parent_id, id from tiles
                where step_id <= {step_id}
                    and parent_id in 
                        (select id from results 
                         where inactivation_step={MAX_STEP} 
                            or completion_step={MAX_STEP})
            """,
            quiet=True,
        )
        if len(tiles_with_active_or_incomplete_parents) > 0:
            raise ValueError(
                f"tiles with active parents: {tiles_with_active_or_incomplete_parents}"
            )

        # we want to ignore tiles that were never active (i.e. pruned during refinement)
        # to do this, we check that the done.step_id is greater than tiles.step_id
        inactive_tiles_with_no_children = self.query(
            f"""
            select id from results
                where 
                    results.step_id <= {step_id}
                    and results.inactivation_step <= {step_id}
                    and id not in (
                        select parent_id from tiles where step_id <= {step_id}
                    )
            """,
            quiet=True,
        )
        if len(inactive_tiles_with_no_children) > 0:
            raise ValueError(
                f"inactive tiles with no children: {inactive_tiles_with_no_children}"
            )

        refined_tiles_with_incorrect_child_count = self.query(
            f"""
            select d.id, count(*) as n_children, max(refine) as n_expected
                from done d
                left join tiles t
                    on t.parent_id = d.id
                where d.step_id <= {step_id}
                    and refine > 0
                group by d.id
                having count(*) != max(refine)
            """,
            quiet=True,
        )
        if len(refined_tiles_with_incorrect_child_count) > 0:
            raise ValueError(
                "refined tiles with wrong number of children:"
                f" {refined_tiles_with_incorrect_child_count}"
            )

        deepened_tiles_with_incorrect_child_count = self.query(
            f"""
            select d.id, count(*) from done d
                left join tiles t
                    on t.parent_id = d.id
                where d.step_id <= {step_id}
                    and deepen=true
                group by d.id
                having count(*) != 1
            """,
            quiet=True,
        )
        if len(deepened_tiles_with_incorrect_child_count) > 0:
            raise ValueError(
                "deepened tiles with wrong number of children:"
                f" {deepened_tiles_with_incorrect_child_count}"
            )

    async def finalize(self) -> None:
        pass


@dataclass
class DuckDBTiles(SQLTiles):
    """
    A tile database built on top of DuckDB.

    See this GitHub issue for a discussion of the design:
    https://github.com/Confirm-Solutions/confirmasaurus/issues/95
    """

    con: "duckdb.DuckDBPyConnection"

    def query(self, query, quiet=False):
        return self.con.query(query).df()

    def insert(self, table, df):
        if not self.does_table_exist(table):
            self.con.execute(f"create table {table} as select * from df")
        else:
            cols = self.con.query(f"select * from {table} limit 0").df().columns
            col_order = ",".join([c for c in cols])
            self.con.execute(f"insert into {table} select {col_order} from df")

    def get_columns(self, table):
        return self.query(f"select * from {table} limit 0").columns

    def does_table_exist(self, table_name):
        out = self.con.query(
            f"""
            select name from sqlite_master 
                where type='table' 
                and name='{table_name}'
            """
        ).fetchall()
        if len(out) == 0:
            return False
        return True

    def insert_done_update_results(self, df: pd.DataFrame):
        self.insert("done", df)
        self.con.execute(
            """
            update results
                set inactivation_step=(
                        CASE WHEN df.active 
                            THEN results.inactivation_step 
                            ELSE df.step_id 
                        END),
                    completion_step=df.step_id
            from df
                where results.id=df.id
            """
        )

    @staticmethod
    def connect(path: str = ":memory:"):
        """
        Load a tile database from a file.

        Args:
            path: The filepath to the database.

        Returns:
            The tile database.
        """
        import duckdb

        return DuckDBTiles(duckdb.connect(path))
