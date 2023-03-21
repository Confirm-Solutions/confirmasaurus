import logging.handlers
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

import pandas as pd

import confirm.adagrid.json as json


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
        assert hasattr(self.db, "insert_logs")
        self.capacity = capacity
        self.flushLevel = flushLevel
        self.interval = interval
        self.lastFlush = time.time()
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
        )

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
            self.db.insert_logs(df)
            self.buffer = []
        finally:
            self.lastFlush = time.time()
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
                not_yet_simulated_df[["zone_id", "step_id", "packet_id"]]
                .drop_duplicates()
                .sort_values(by=["zone_id", "step_id", "packet_id"])
                .values,
            )
        )

    def get_zone_steps(self):
        return self.tiles.groupby("zone_id")["step_id"].max().to_dict()

    def n_existing_packets(self, zone_id, step_id):
        return (
            self.tiles[
                (self.tiles["zone_id"] == zone_id) & (self.tiles["step_id"] == step_id)
            ]["packet_id"].max()
            + 1
        )

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
        self.tiles.loc[df["id"], "active"] = df["active"]
        self.results.loc[df["id"], "eligible"] = False
        self.results.loc[df["id"], "active"] = df["active"]

    def get_packet(self, zone_id: int, step_id: int, packet_id: int) -> pd.DataFrame:
        where = (self.tiles["step_id"] == step_id) & (
            self.tiles["packet_id"] == packet_id
        )
        return self.tiles.loc[where]

    def next(
        self, zone_id: int, new_step_id: int, n: int, order_col: str
    ) -> pd.DataFrame:
        out = self.results.loc[self.results["eligible"]].nsmallest(n, order_col)
        return out

    def bootstrap_lamss(self, zone_id: int) -> pd.Series:
        nB = (
            max([int(c[6:]) for c in self.results.columns if c.startswith("B_lams")])
            + 1
        )
        active_tiles = self.results.loc[self.results["active"]]
        return active_tiles[[f"B_lams{i}" for i in range(nB)]].values.min(axis=0)

    def worst_tile(self, zone_id: int, orderer: str) -> pd.DataFrame:
        active_tiles = self.results.loc[self.results["active"]]
        return active_tiles.loc[[active_tiles[orderer].idxmin()]]

    async def verify(self):
        pass

    async def init_grid(self, df: pd.DataFrame) -> None:
        df = df.set_index("id")
        df.insert(0, "id", df.index)
        self.tiles = df


@dataclass
class DuckDBTiles:
    """
    A tile database built on top of DuckDB. This should be very fast and
    robust and is the default database for confirm.

    See this GitHub issue for a discussion of the design:
    https://github.com/Confirm-Solutions/confirmasaurus/issues/95
    """

    con: "duckdb.DuckDBPyConnection"
    _tiles_columns_cache: List[str] = None
    _results_columns_cache: List[str] = None
    _d: int = None

    def __post_init__(self):
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

    def dimension(self):
        if self._d is None:
            cols = self._tiles_columns()
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
        return self.con.execute("select * from results").df()

    def get_done(self):
        return self.con.execute("select * from done").df()

    def get_reports(self):
        json_strs = self.con.execute("select * from reports").fetchall()
        return pd.DataFrame([json.loads(s[0]) for s in json_strs])

    def get_incomplete_packets(self):
        if self.does_table_exist("results"):
            restrict = "and id not in (select id from results)"
        else:
            restrict = ""
        return self.con.query(
            f"""
            select zone_id, step_id, packet_id
                from tiles
                where active=true {restrict}
                group by zone_id, step_id, packet_id
                order by zone_id, step_id, packet_id
            """
        ).fetchall()

    def get_zone_steps(self):
        return dict(
            self.con.query(
                """
            select zone_id, max(step_id)
                from tiles
                    where coordination_id = (select max(coordination_id) from tiles)
                group by zone_id
                order by zone_id
            """
            ).fetchall()
        )

    def n_existing_packets(self, zone_id, step_id):
        return self.con.query(
            f"""
            select max(packet_id) + 1 from tiles
                where zone_id = {zone_id} and step_id = {step_id}
            """
        ).fetchone()[0]

    def insert_report(self, report):
        self.con.execute(f"insert into reports values ('{json.dumps(report)}')")
        return report

    def insert_tiles(self, df: pd.DataFrame):
        column_order = ",".join(self._tiles_columns())
        self.con.execute(f"insert into tiles select {column_order} from df")

    def insert_results(self, df: pd.DataFrame, orderer: str):
        if not self.does_table_exist("results"):
            self.con.execute("create table if not exists results as select * from df")
            return
        column_order = ",".join(self._results_columns())
        self.con.execute(f"insert into results select {column_order} from df")

    def insert_done(self, which):
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

    def get_packet(self, zone_id: int, step_id: int, packet_id: int = None):
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
                    zone_id = {zone_id}
                    and step_id = {step_id}
                    and active=true
                    {restrict_clause}
            """,
        ).df()

    def next(
        self, zone_id: int, new_step_id: int, n: int, orderer: str
    ) -> pd.DataFrame:
        # we wrap with a transaction to ensure that concurrent readers don't
        # grab the same chunk of work.
        t = self.con.begin()
        out = t.execute(
            f"""
            select * from results 
                where eligible=true
                    and zone_id = {zone_id}
                    and step_id < {new_step_id}
            order by {orderer} limit {n}
            """
        ).df()
        t.commit()
        return out

    def bootstrap_lamss(self, zone_id: int) -> List[float]:
        # Get the number of bootstrap lambda* columns
        nB = (
            max([int(c[6:]) for c in self._results_columns() if c.startswith("B_lams")])
            + 1
        )

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        if zone_id is None:
            zone_id_clause = ""
        else:
            zone_id_clause = f"and zone_id = {zone_id}"
        lamss = self.con.execute(
            f"""
            select {cols} from results 
                where active=true
                    {zone_id_clause}
            """
        ).fetchall()[0]

        return lamss

    def worst_tile(self, zone_id, order_col):
        if zone_id is None:
            zone_id_clause = ""
        else:
            zone_id_clause = f"and zone_id = {zone_id}"
        return self.con.execute(
            f"""
            select * from results
                where active=true
                    {zone_id_clause}
                order by {order_col} limit 1
            """
        ).df()

    def coordinate(self, step_id: int, n_zones: int):
        if not self.does_table_exist("zone_mapping"):
            self.con.execute(
                """
                create table zone_mapping (
                    id ubigint,
                    coordination_id int,
                    old_zone_id int,
                    new_zone_id int,
                    before_step_id int
                )"""
            )

        coordination_id = self.con.query(
            """
            select max(coordination_id) + 1 from results
            """
        ).fetchone()[0]

        ordering = ",".join(
            [f"theta{i}" for i in range(self.dimension())]
            + [c for c in self._results_columns() if c.startswith("null_truth")]
        )
        self.con.execute(
            f"""
            insert into zone_mapping 
                select id, {coordination_id}, zone_id,
                        (row_number() OVER (order by {ordering}))%{n_zones},
                        {step_id}
                    from results
                        where eligible=true
                            and active=true
                    order by {ordering}
        """
        )
        self.con.execute(
            f"""
            update results set 
                    zone_id=(
                        select new_zone_id from zone_mapping
                            where zone_mapping.id=results.id
                            and zone_mapping.coordination_id={coordination_id}
                    ),
                    coordination_id={coordination_id}
                where eligible=true
                    and active=true
        """
        )
        n_results = self.con.query(
            f""" 
            select count(*) from results where coordination_id={coordination_id}
        """
        ).fetchone()[0]
        zone_steps = dict(
            self.con.query(
                f"""
            select zone_id, max(step_id) from results
                    where coordination_id={coordination_id}
                group by zone_id
                order by zone_id
        """
            ).fetchall()
        )
        return n_results, zone_steps

    def get_zone_mapping(self):
        return self.con.execute("select * from zone_mapping").df()

    async def verify(db):
        duplicate_tiles = db.con.query(
            "select id from tiles group by id having count(*) > 1"
        ).df()
        if len(duplicate_tiles) > 0:
            raise ValueError(f"Duplicate tiles: {duplicate_tiles}")

        duplicate_results = db.con.query(
            "select id from results group by id having count(*) > 1"
        ).df()
        if len(duplicate_results) > 0:
            raise ValueError(f"Duplicate results: {duplicate_results}")

        duplicate_done = db.con.query(
            "select id from done group by id having count(*) > 1"
        ).df()
        if len(duplicate_done) > 0:
            raise ValueError(f"Duplicate done: {duplicate_done}")

        results_without_tiles = db.con.query(
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

        tiles_without_results = db.con.query(
            """
            select id from tiles
            -- packet_id >= 0 excludes tiles that were split or pruned
                where packet_id >= 0
                    and id not in (select id from results)
            """
        ).df()
        if len(tiles_without_results) > 0:
            raise ValueError(
                "Rows in tiles without corresponding rows in results:"
                f" {tiles_without_results}"
            )

        tiles_without_parents = db.con.query(
            """
            select parent_id, id from tiles
                where parent_id not in (select id from done)
            """
        ).df()
        if len(tiles_without_parents) > 0:
            raise ValueError(f"tiles without parents: {tiles_without_parents}")

        tiles_with_active_or_eligible_parents = db.con.query(
            """
            select parent_id, id from tiles
                where parent_id in 
                    (select id from results where active=true or eligible=true)
            """
        ).df()
        if len(tiles_with_active_or_eligible_parents) > 0:
            raise ValueError(
                f"tiles with active parents: {tiles_with_active_or_eligible_parents}"
            )

        inactive_tiles_with_no_children = db.con.query(
            """
            select id from tiles
            -- packet_id >= 0 excludes tiles that were split or pruned
                where packet_id >= 0
                    and active=false
                    and id not in (select parent_id from tiles)
            """
        ).df()
        if len(inactive_tiles_with_no_children) > 0:
            raise ValueError(
                f"inactive tiles with no children: {inactive_tiles_with_no_children}"
            )

        refined_tiles_with_incorrect_child_count = db.con.query(
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

        deepened_tiles_with_incorrect_child_count = db.con.query(
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

    def insert_logs(self, df):
        self.con.execute("insert into logs select * from df")

    def get_logs(self):
        return self.con.execute("select * from logs").df()

    def close(self) -> None:
        self.con.close()

    def get_null_hypos(self):
        return self.con.query("select * from null_hypos").df()

    def get_config(self):
        return self.con.query("select * from config").df()

    def insert_config(self, cfg_df):
        return self.con.execute("insert into config select * from cfg_df")

    async def init_grid(
        self, tiles_df: pd.DataFrame, null_hypos_df: pd.DataFrame, cfg_df: pd.DataFrame
    ) -> None:
        self.con.execute("create table tiles as select * from tiles_df")
        self.con.execute(
            """
            create table done (
                    zone_id UINTEGER,
                    step_id UINTEGER,
                    packet_id INTEGER,
                    id UBIGINT,
                    active BOOL,
                    finisher_id UINTEGER,
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

    @staticmethod
    def connect(path=":memory:"):
        """
        Load a tile database from a file.

        Args:
            path: The filepath to the database.

        Returns:
            The tile database.
        """
        import duckdb

        return DuckDBTiles(duckdb.connect(path))


done_cols = [
    "zone_id",
    "step_id",
    "packet_id",
    "id",
    "active",
    "finisher_id",
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
