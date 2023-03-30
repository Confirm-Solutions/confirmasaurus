import asyncio
import logging.handlers
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

import pandas as pd

import confirm.adagrid.json as json
from .const import MAX_STEP

if TYPE_CHECKING:
    import duckdb
    import clickhouse_connect

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
        assert hasattr(self.db, "insert_logs")
        self.capacity = capacity
        self.flushLevel = flushLevel
        self.interval = interval
        self.lastFlush = time.time()
        self.creating_thread = threading.current_thread().ident
        super().__init__(self.capacity)

    def __enter__(self):
        self.lastFlush = time.time()
        logging.getLogger().addHandler(self)

    def __exit__(self, *_):
        self.close()
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
            self.db.insert_logs(df)
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

    def get_incomplete_packets(self):
        if self.results is None:
            not_yet_simulated_df = self.tiles
        else:
            joined_df = self.tiles.set_index("id").merge(
                self.results[["id"]].set_index("id"),
                on="id",
                how="left",
                indicator=True,
            )
            not_yet_simulated_df = self.tiles[joined_df["_merge"] == "left_only"]

        return list(
            map(
                tuple,
                not_yet_simulated_df[["step_id", "packet_id"]]
                .drop_duplicates()
                .sort_values(by=["step_id", "packet_id"])
                .values,
            )
        )

    def n_existing_packets(self, step_id):
        return self.tiles[self.tiles["step_id"] == step_id]["packet_id"].max() + 1

    def insert_report(self, report):
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

    def get_packet(self, step_id: int, packet_id: int) -> pd.DataFrame:
        where = (self.tiles["step_id"] == step_id) & (
            self.tiles["packet_id"] == packet_id
        )
        return self.tiles.loc[where]

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
class DuckDBTiles:
    """
    A tile database built on top of DuckDB.

    See this GitHub issue for a discussion of the design:
    https://github.com/Confirm-Solutions/confirmasaurus/issues/95

    See this github issue for thoughts on the mirrored insert backup to Clickhouse:
    https://github.com/Confirm-Solutions/confirmasaurus/issues/323
    """

    con: "duckdb.DuckDBPyConnection"
    ch_client: "clickhouse_connect.driver.httpclient.HttpClient" = None
    ch_table_exists: set = field(default_factory=set)
    ch_tasks: List[asyncio.Task] = field(default_factory=list)
    _d: int = None

    def dimension(self):
        if self._d is None:
            cols = self._tiles_columns()
            self._d = max([int(c[5:]) for c in cols if c.startswith("theta")]) + 1
        return self._d

    def _tiles_columns(self):
        return self.con.execute("select * from tiles limit 0").df().columns

    def _results_columns(self):
        return self.con.execute("select * from results limit 0").df().columns

    def _done_columns(self):
        return self.con.execute("select * from done limit 0").df().columns

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

    def get_tiles(self):
        return self.con.execute("select * from tiles").df()

    def get_results(self):
        out = self.con.execute("select * from results").df()
        out.insert(0, "active", out["inactivation_step"] == MAX_STEP)
        return out

    def get_done(self):
        return self.con.execute("select * from done").df()

    def get_reports(self):
        json_strs = self.con.execute("select * from reports").fetchall()
        return pd.DataFrame([json.loads(s[0]) for s in json_strs])

    def get_logs(self):
        return self.con.execute("select * from logs").df()

    def get_null_hypos(self):
        return self.con.query("select * from null_hypos").df()

    def get_config(self):
        return self.con.query("select * from config").df()

    def get_incomplete_packets(self):
        if self.does_table_exist("results"):
            restrict = "and id not in (select id from results)"
        else:
            restrict = ""
        return self.con.query(
            f"""
            select step_id, packet_id
                from tiles
                where inactivation_step={MAX_STEP} {restrict}
                group by step_id, packet_id
                order by step_id, packet_id
            """
        ).fetchall()

    def n_existing_packets(self, step_id):
        return self.con.query(
            f"""
            select max(packet_id) + 1 from tiles
                where step_id = {step_id}
            """
        ).fetchone()[0]

    def insert_report(self, report):
        report_str = json.dumps(report)
        self.con.execute(f"insert into reports values ('{report_str}')")
        self.ch_insert("reports", pd.DataFrame(dict(json=[report_str])), create=False)
        return report

    def insert_tiles(self, df: pd.DataFrame, ch_insert: bool = False):
        # NOTE: We insert to Clickhouse tiles and results in the packet
        # processing instead. This spreads out the Clickhouse mirroring
        # bandwidth requirements.
        column_order = ",".join(self._tiles_columns())
        self.con.execute(f"insert into tiles select {column_order} from df")
        logger.debug("Inserted %d new tiles.", df.shape[0])
        if ch_insert:
            self.ch_insert("tiles", df, create=False)

    def insert_results(self, df: pd.DataFrame, orderer: str):
        # NOTE: We insert to Clickhouse tiles and results in the packet
        # processing instead. This spreads out the Clickhouse mirroring
        # bandwidth requirements.
        df_cols = ",".join([f"df.{c}" for c in df.columns if c != "id"])
        if not self.does_table_exist("results"):
            self.con.execute(
                f"""
                create table results as (
                    select tiles.*, {df_cols} from df
                    left join tiles on df.id = tiles.id
                )
            """
            )
        else:
            self.con.execute(
                f"""
                insert into results select tiles.*, {df_cols} from df
                    left join tiles on df.id = tiles.id
            """
            )
        logger.debug(f"Inserted {df.shape[0]} results.")

    def insert_done(self, df: pd.DataFrame):
        column_order = ",".join(self._done_columns())
        self.con.execute(f"insert into done select {column_order} from df")
        self.ch_insert("done", df, create=False)
        self.update_active_complete(df=df)
        # Updating active/complete is not strictly necessary since it can be
        # inferred from the info in the done table. But, updating the flags on
        # the tiles/results tables is more efficient for future queries.
        logger.debug(f"Finished {df.shape[0]} tiles.")

    def update_active_complete(self, df=None):
        if df is None:
            table = "done"
        else:
            table = "df"
        self.con.execute(
            f"""
            update tiles 
                set inactivation_step=d.step_id
            from {table} d
                where tiles.id=d.id
                    and d.active=false
            """
        )
        self.con.execute(
            f"""
            update results
                set inactivation_step=d.step_id
            from {table} d
                where results.id=d.id
                    and d.active=false
            """
        )
        self.con.execute(
            f"""
            update results
                set completion_step=d.step_id
            from {table} d
                where results.id=d.id
            """
        )

    def insert_logs(self, df):
        if not self.does_table_exist("logs"):
            self.con.execute(
                """
                create table if not exists logs (
                    t TIMESTAMP,
                    name TEXT,
                    pathname TEXT,
                    lineno UINTEGER,
                    levelno UINTEGER,
                    levelname TEXT,
                    message TEXT
                )
                """
            )
        self.con.execute("insert into logs select * from df")
        self.ch_insert("logs", df, create=True)

    def insert_config(self, cfg_df):
        self.con.execute("insert into config select * from cfg_df")
        self.ch_insert("config", cfg_df, create=False)

    def get_packet(self, step_id: int, packet_id: int = None):
        if self.does_table_exist("results"):
            restrict_clause = "and id not in (select id from results)"
        else:
            restrict_clause = ""
        if packet_id is not None:
            restrict_clause += f"and packet_id = {packet_id}"
        return self.con.execute(
            f"""
            select * from tiles
                where
                    step_id = {step_id}
                    and inactivation_step > {step_id}
                    {restrict_clause}
            """,
        ).df()

    def next(
        self, basal_step_id: int, new_step_id: int, n: int, orderer: str
    ) -> pd.DataFrame:
        # we wrap with a transaction to ensure that concurrent readers don't
        # grab the same chunk of work.
        t = self.con.begin()
        out = t.execute(
            f"""
            select * from results 
                where completion_step >= {new_step_id}
                    and step_id <= {basal_step_id}
            order by {orderer} limit {n}
            """
        ).df()
        t.commit()
        return out

    def bootstrap_lamss(self, basal_step_id: int) -> List[float]:
        # Get the number of bootstrap lambda* columns
        nB = (
            max([int(c[6:]) for c in self._results_columns() if c.startswith("B_lams")])
            + 1
        )

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        lamss = self.con.execute(
            f"""
            select {cols} from results 
                where inactivation_step > {basal_step_id}
                    and step_id <= {basal_step_id}
            """
        ).fetchall()[0]

        return lamss

    def worst_tile(self, basal_step_id, order_col):
        return self.con.execute(
            f"""
            select * from results
                where inactivation_step > {basal_step_id}
                    and step_id <= {basal_step_id}
                order by {order_col} limit 1
            """
        ).df()

    def get_next_step(self):
        return self.con.query("select max(step_id) + 1 from tiles").fetchone()[0]

    def verify(self, step_id: int):
        duplicate_tiles = self.con.query(
            "select id from tiles group by id having count(*) > 1"
        ).df()
        if len(duplicate_tiles) > 0:
            raise ValueError(f"Duplicate tiles: {duplicate_tiles}")

        duplicate_results = self.con.query(
            "select id from results group by id having count(*) > 1"
        ).df()
        if len(duplicate_results) > 0:
            raise ValueError(f"Duplicate results: {duplicate_results}")

        duplicate_done = self.con.query(
            "select id from done group by id having count(*) > 1"
        ).df()
        if len(duplicate_done) > 0:
            raise ValueError(f"Duplicate done: {duplicate_done}")

        results_without_tiles = self.con.query(
            """
            select id from results
                where id not in (select id from tiles)
            """
        ).df()
        if len(results_without_tiles) > 0:
            raise ValueError(
                "Rows in results without corresponding rows in tiles:"
                f" {results_without_tiles}"
            )

        tiles_without_results = self.con.query(
            f"""
            select id from tiles
            -- packet_id >= 0 excludes tiles that were split or pruned
                where packet_id >= 0
                    and step_id <= {step_id}
                    and id not in (select id from results)
            """
        ).df()
        if len(tiles_without_results) > 0:
            raise ValueError(
                "Rows in tiles without corresponding rows in results:"
                f" {tiles_without_results}"
            )

        tiles_without_parents = self.con.query(
            """
            select parent_id, id from tiles
                where parent_id not in (select id from done)
            """
        ).df()
        if len(tiles_without_parents) > 0:
            raise ValueError(f"tiles without parents: {tiles_without_parents}")

        tiles_with_active_or_incomplete_parents = self.con.query(
            f"""
            select parent_id, id from tiles
                where parent_id in 
                    (select id from results 
                         where inactivation_step={MAX_STEP} 
                            or completion_step={MAX_STEP})
            """
        ).df()
        if len(tiles_with_active_or_incomplete_parents) > 0:
            raise ValueError(
                f"tiles with active parents: {tiles_with_active_or_incomplete_parents}"
            )

        inactive_tiles_with_no_children = self.con.query(
            f"""
            select id from tiles
            -- packet_id >= 0 excludes tiles that were split or pruned
                where packet_id >= 0
                    and inactivation_step < {MAX_STEP}
                    and id not in (select parent_id from tiles)
            """
        ).df()
        if len(inactive_tiles_with_no_children) > 0:
            raise ValueError(
                f"inactive tiles with no children: {inactive_tiles_with_no_children}"
            )

        refined_tiles_with_incorrect_child_count = self.con.query(
            """
            select d.id, count(*) as n_children, max(refine) as n_expected
                from done d
                left join tiles t
                    on t.parent_id = d.id
                where refine > 0
                group by d.id
                having count(*) != max(refine)
            """
        ).df()
        if len(refined_tiles_with_incorrect_child_count) > 0:
            raise ValueError(
                "refined tiles with wrong number of children:"
                f" {refined_tiles_with_incorrect_child_count}"
            )

        deepened_tiles_with_incorrect_child_count = self.con.query(
            """
            select d.id, count(*) from done d
                left join tiles t
                    on t.parent_id = d.id
                where deepen=true
                group by d.id
                having count(*) != 1
            """
        ).df()
        if len(deepened_tiles_with_incorrect_child_count) > 0:
            raise ValueError(
                "deepened tiles with wrong number of children:"
                f" {deepened_tiles_with_incorrect_child_count}"
            )

    def close(self) -> None:
        self.con.close()

    def init_grid(
        self, tiles_df: pd.DataFrame, null_hypos_df: pd.DataFrame, cfg_df: pd.DataFrame
    ) -> None:
        self.con.execute("create table tiles as select * from tiles_df")
        self.con.execute(
            """
            create table done (
                    step_id UINTEGER,
                    packet_id INTEGER,
                    id UBIGINT,
                    active BOOL,
                    refine UINTEGER,
                    deepen UINTEGER,
                    split BOOL)
            """
        )
        absent_parents_df = get_absent_parents(tiles_df)  # noqa
        self.con.execute("insert into done select * from absent_parents_df")
        self.con.execute("create table reports (json TEXT)")
        self.con.execute("create table null_hypos as select * from null_hypos_df")
        self.con.execute("create table config as select * from cfg_df")
        self.ch_insert("tiles", tiles_df, create=True)
        self.ch_insert("done", absent_parents_df, create=True)
        self.ch_insert("null_hypos", null_hypos_df, create=True)
        self.ch_insert("config", cfg_df, create=True)
        self.ch_insert("reports", pd.DataFrame(columns=["json"]), create=True)

    @staticmethod
    def connect(path: str = ":memory:", ch_client: Optional[str] = None):
        """
        Load a tile database from a file.

        Args:
            path: The filepath to the database.
            ch_client: A Clickhouse client to use for mirroring inserts.

        Returns:
            The tile database.
        """
        import duckdb

        return DuckDBTiles(duckdb.connect(path), ch_client)

    def ch_insert(self, table: str, df: pd.DataFrame, create: bool):
        if self.ch_client is not None:
            import confirm.cloud.clickhouse as ch

            if create and table not in self.ch_table_exists:
                ch.create_table(self.ch_client, table, df)
                self.ch_table_exists.add(table)
            self.ch_tasks.extend(ch.threaded_block_insert_df(self.ch_client, table, df))

    async def ch_wait(self):
        while len(self.ch_tasks) > 0:
            tmp = self.ch_tasks
            self.ch_tasks = []
            await asyncio.gather(*tmp)


done_cols = [
    "step_id",
    "packet_id",
    "id",
    "active",
    "refine",
    "deepen",
    "split",
]


def get_absent_parents(tiles_df):
    # these tiles have no parents. poor sad tiles :(
    # we need to put these absent parents into the done table
    absent_parents = pd.DataFrame(
        tiles_df["parent_id"].unique()[:, None], columns=["id"]
    )
    for c in done_cols:
        if c not in absent_parents.columns:
            absent_parents[c] = 0
    return absent_parents[done_cols]
