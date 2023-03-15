import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow

import confirm.adagrid.json as json
from confirm.adagrid.db import get_absent_parents

if TYPE_CHECKING:
    import clickhouse_connect

import logging

logger = logging.getLogger(__name__)

all_tables = [
    "results",
    "tiles",
    "done",
    "config",
    "logs",
    "reports",
    "null_hypos",
    "zone_mapping",
]


def backup(duck, ch_db):
    for name in all_tables:
        backup_table(duck, ch_db, name)


def restore(duck, ch_db):
    for name in all_tables:
        restore_table(duck, ch_db, name)


def backup_table(duck, ch_db, name):
    if not duck.does_table_exist(name):
        logger.info(
            f"Backup skipping table {name} because it"
            " doesn't exist in the source db."
        )
        return
    df = duck.con.query(f"select * from {name}").df()
    cols = get_create_table_cols(df)
    if ch_db.does_table_exist(name):
        _command(ch_db.client, f"DROP TABLE {name}")
    _command(
        ch_db.client,
        f"""
        CREATE TABLE {name} ({",".join(cols)})
        ENGINE = MergeTree()
        ORDER BY ()
        """,
    )
    _insert_df(ch_db.client, name, df)


def restore_table(duck, ch_db, name):
    if not ch_db.does_table_exist(name):
        logger.info(
            f"Restore skipping table {name} because it"
            " doesn't exist in the source db."
        )
        return
    df = _query_df(ch_db.client, f"select * from {name}")
    if name == "logs":
        df["t"] = df["t"].dt.tz_localize(None)
    if duck.does_table_exist(name):
        duck.con.execute(f"drop table {name}")
    duck.con.execute(f"create table {name} as select * from df")


type_map = {
    "uint8": "UInt8",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
    "int32": "Int32",
    "int64": "Int64",
    "bool": "Boolean",
    "string": "String",
    "object": "String",
    "datetime64[ns]": "DateTime64(9)",
}


def get_create_table_cols(df):
    """
    Map from Pandas dtypes to Clickhouse types.

    Args:
        df: The dataframe

    Returns:
        A list of strings of the form "column_name type"
    """
    return [
        f"{c} Nullable({type_map[dt.name]})" for c, dt in zip(df.columns, df.dtypes)
    ]


def retry_ch_action(method, *args, retries=2, is_retry=False, **kwargs):
    import clickhouse_connect

    if is_retry:
        logger.error(f"Retrying {method.__name__} with {args} {kwargs}")

    try:
        return method(*args, **kwargs)
    except clickhouse_connect.driver.exceptions.DatabaseError as e:
        msg = e.args[0]
        # https://github.com/Confirm-Solutions/confirmasaurus/actions/runs/4310150027/jobs/7518302759 # noqa
        # Code: 319. DB::Exception: Unknown status, client must retry.
        # Reason: Code: 254. DB::Exception: Replica become inactive while
        # waiting for quorum. (NO_ACTIVE_REPLICAS)
        if "Code: 319" in msg:
            return retry_ch_action(
                method, *args, retries=retries - 1, is_retry=True, **kwargs
            )

        # https://github.com/Confirm-Solutions/confirmasaurus/actions/runs/4310010650/jobs/7518016908 # noqa
        # Code: 285. DB::Exception: Number of alive replicas (1) is less than
        # requested quorum (2/2). (TOO_FEW_LIVE_REPLICAS)
        if "Code: 285" in msg:
            return retry_ch_action(
                method, *args, retries=retries - 1, is_retry=True, **kwargs
            )

        raise e

    # https://github.com/Confirm-Solutions/confirmasaurus/actions/runs/4305452673/jobs/7507911386 # noqa
    # clickhouse_connect.driver.exceptions.OperationalError: Error
    # HTTPSConnectionPool(...) Read timed out. (read timeout=300)
    # executing HTTP request
    except clickhouse_connect.driver.exceptions.OperationalError as e:
        msg = e.args[0]
        if "Read timed out" in msg:
            return retry_ch_action(
                method, *args, retries=retries - 1, is_retry=True, **kwargs
            )

        raise e


def _query_df(client, query):
    # Loading via Arrow and then converting to Pandas is faster than using
    # query_df directly to load a pandas dataframe. I'm guessing that loading
    # through arrow is a little less flexible or something, but for our
    # purposes, faster loading is great.
    start = time.time()
    out = retry_ch_action(client.query_arrow, query).to_pandas()
    logger.debug(f"Query took {time.time() - start} seconds\n{query}")
    return out


default_insert_settings = dict(
    insert_distributed_sync=1, insert_quorum="auto", insert_quorum_parallel=1
)
# default_async_insert_settings = {"async_insert": 1, "wait_for_async_insert": 0}


def _insert_df(client, table, df, settings=None):
    # Same as _query_df, inserting through arrow is faster!
    start = time.time()
    retry_ch_action(
        client.insert_arrow,
        table,
        pyarrow.Table.from_pandas(df, preserve_index=False),
        settings=settings or default_insert_settings,
    )
    logger.debug(
        f"Inserting {df.shape[0]} rows into {table} took"
        f" {time.time() - start} seconds"
    )


def _query(client, query, *args, **kwargs):
    start = time.time()
    out = retry_ch_action(client.query, query, *args, **kwargs)
    logger.debug(f"Query took {time.time() - start} seconds\n{query} ")
    return out


def _command(client, query, *args, **kwargs):
    start = time.time()
    out = retry_ch_action(client.command, query, *args, **kwargs)
    logger.debug(f"Command took {time.time() - start} seconds\n{query}")
    return out


@dataclass
class Clickhouse:
    """
    A tile database built on top of Clickhouse. This should be very fast and
    robust and is preferred for large runs. Latency will be worse than with
    DuckDB because network requests will be required for each query.

    See the DuckDBTiles or PandasTiles implementations for details on the
    Adagrid tile database interface.

    The active and eligible tile columns are stored across two places in order
    to construct an append-only table structure.

    active: True if the tile is a leaf in the tile tree. Specified in two ways
        - the "active" column in the tiles table
        - the "done" table, which contains a column indicating which
          tiles are inactive.
        - a tile is only active is active=True and id not in inactive

    eligible: True if the tile is eligible to be selected for work. Specified in
         two ways
        - the "eligibile" column in the tiles table
        - the "tiles_work" table, which contains tiles that are ineligible
          because they either are being worked on or have been worked on
        - a tile is only eligible if eligible=True and id not in tiles_work

    The reason for these dual sources of information is the desire to only
    rarely update rows that have already been written. Updating data is painful
    in Clickhouse. We could periodically flush the tiles_ineligible and
    inactive tables to corresponding columns in the tiles table. This
    would reduce the size of the subqueries and probably improve performance
    for very large jobs. But, this flushing would probably only need to happen
    once every few hours.

    Packets: A packet is a distributed unit of work. We check the convergence
    criterion once an entire packet of work is done. This removes data race
    conditions where one worker might finish work before or after another thus
    affecting the convergence criterion and tile selection process.

    NOTE: A tile may be active but ineligible if it was selected for work but the
    criterion decided it was not worth refining or deepening.

    NOTE: We could explore a Clickhouse projection or a Clickhouse materialized
    view as alternative approaches to the dual sources of information for
    active and eligible tiles.
    """

    connection_details: Dict[str, str]
    client: "clickhouse_connect.driver.httpclient.HttpClient"
    job_id: str
    _tiles_columns_cache: List[str] = None
    _results_columns_cache: List[str] = None
    _d: int = None
    _results_table_exists: bool = False
    is_distributed: bool = True
    supports_threads: bool = True

    def dimension(self):
        if self._d is None:
            cols = self._tiles_columns()
            self._d = max([int(c[5:]) for c in cols if c.startswith("theta")]) + 1
        return self._d

    def _tiles_columns(self):
        if self._tiles_columns_cache is None:
            self._tiles_columns_cache = _query_df(
                self.client, "select * from tiles limit 1"
            ).columns
        return self._tiles_columns_cache

    def _results_columns(self):
        if self._results_columns_cache is None:
            self._results_columns_cache = _query_df(
                self.client, "select * from results limit 1"
            ).columns
        return self._results_columns_cache

    def does_table_exist(self, table_name: str) -> bool:
        return (
            len(
                _query(
                    self.client,
                    f"""
            select * from information_schema.tables
                where table_schema = '{self.job_id}'
                    and table_name = '{table_name}'
            """,
                ).result_set
            )
            > 0
        )

    def results_table_exists(self):
        if not self._results_table_exists:
            self._results_table_exists = self.does_table_exist("results")
            if not self._results_table_exists:
                return False
        return True

    def get_tiles(self):
        cols = ",".join(
            [c for c in self._tiles_columns() if c not in ["active", "eligible"]]
        )
        return _query_df(
            self.client,
            f"""
            select {cols},
                and(active=true,
                    (id not in (select id from done where active=false)))
                    as active
            from tiles
        """,
        )

    def get_results(self):
        cols = ",".join(
            [c for c in self._results_columns() if c not in ["active", "eligible"]]
        )
        out = _query_df(
            self.client,
            f"""
            select {cols},
                and(active=true,
                    (id not in (select id from done where active=false)))
                    as active,
                and(eligible=true,
                    (id not in (select id from done)))
                    as eligible
            from results
        """,
        )
        out["active"] = out["active"].astype(bool)
        out["eligible"] = out["eligible"].astype(bool)
        return out

    def get_done(self):
        return _query_df(self.client, "select * from done")

    def get_reports(self):
        json_strs = _query(self.client, "select * from reports").result_set
        return pd.DataFrame([json.loads(s[0]) for s in json_strs]).sort_values(
            by=["zone_id", "step_id", "packet_id"]
        )

    def get_incomplete_packets(self):
        if self.results_table_exists():
            restrict = "where id not in (select id from results)"
        else:
            restrict = ""
        return _query(
            self.client,
            f"""
            select zone_id, step_id, packet_id
                from tiles {restrict}
                group by zone_id, step_id, packet_id
                order by zone_id, step_id, packet_id
            """,
        ).result_set

    def get_zone_steps(self):
        if self.does_table_exist("zone_mapping"):
            cte = """
            with (select max(coordination_id) from zone_mapping) 
                as max_coordination_id
            """
            restrict = "where coordination_id = max_coordination_id"
        else:
            cte, restrict = "", ""

        raw = _query(
            self.client,
            f"""
            {cte}
            select zone_id, max(step_id)
                from tiles
                {restrict}
                group by zone_id
                order by zone_id
            """,
        ).result_set
        return dict(raw)

    def n_existing_packets(self, zone_id, step_id):
        rows = _query(
            self.client,
            f"""
            select packet_id from tiles
                where zone_id = {zone_id} and step_id = {step_id}
                order by packet_id desc
                limit 1
        """,
        ).result_set
        if len(rows) == 0:
            return 0
        else:
            return rows[0][0] + 1

    def insert_report(self, report):
        _command(self.client, f"insert into reports values ('{json.dumps(report)}')")
        return report

    def insert_tiles(self, df: pd.DataFrame):
        logger.debug(f"writing {df.shape[0]} tiles")
        _insert_df(self.client, "tiles", df)

    def insert_results(self, df: pd.DataFrame, orderer: str):
        self._create_results_table(df, orderer)
        logger.debug(f"writing {df.shape[0]} results")
        _insert_df(self.client, "results", df)
        self._results_table_exists = True

    def insert_done(self, which):
        logger.debug(f"finishing {which.shape[0]} tiles")
        _insert_df(self.client, "done", which)

    def get_packet(self, zone_id: int, step_id: int, packet_id: int = None):
        """
        `get_packet` is used to select tiles for processing/simulation.
        """
        if self.results_table_exists():
            restrict_clause = "and id not in (select id from results)"
        else:
            restrict_clause = ""
        if packet_id is not None:
            restrict_clause += f"and packet_id = {packet_id}"
        return _query_df(
            self.client,
            f"""
            select * from tiles
                where
                    zone_id = {zone_id}
                    and step_id = {step_id}
                    {restrict_clause}
            """,
        )

    def next(self, zone_id, new_step_id, n, orderer):
        """
        `next` is used in select_tiles to get the next batch of tiles to
        refine/deepen
        """
        return _query_df(
            self.client,
            f"""
            select * from results
                where
                    zone_id = {zone_id}
                    and step_id < {new_step_id}
                    and eligible=true
                    and id not in (select id from done)
            order by {orderer} limit {n}
        """,
        )

    def worst_tile(self, zone_id, orderer):
        if zone_id is None:
            zone_id_clause = ""
        else:
            zone_id_clause = f"and zone_id = {zone_id}"

        return _query_df(
            self.client,
            f"""
            select * from results
                where
                    active=true
                    and id not in (select id from done where active=false)
                    {zone_id_clause}
            order by {orderer} limit 1
        """,
        )

    def bootstrap_lamss(self, zone_id):
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
        lamss = _query(
            self.client,
            f"""
            select {cols} from results where
                active=true
                and id not in (select id from done where active=false)
                {zone_id_clause}
        """,
        ).result_set[0]

        return lamss

    alter_settings = {
        "allow_nondeterministic_mutations": "1",
        "mutations_sync": 2,
    }

    def update_active_eligible(self):
        _command(
            self.client,
            """
            ALTER TABLE results
            UPDATE
                eligible = and(eligible=true, id not in (select id from done)),
                active = and(active=true, 
                    id not in (select id from done where active=false))
            WHERE
                eligible = 1
                and active = 1
            """,
            settings=self.alter_settings,
        )

    def get_active_eligible(self):
        # We need a unique and deterministic ordering for the tiles returned
        # herer. Since we are filtering to active/eligible tiles, there can be
        # no duplicates when sorted by
        # (theta0,...,thetan, null_truth0, ..., null_truthn)
        ordering = ",".join(
            [f"theta{i}" for i in range(self.dimension())]
            + [c for c in self._results_columns() if c.startswith("null_truth")]
        )
        return _query_df(
            self.client,
            f"""
            SELECT * FROM results
            WHERE eligible = 1
                and id not in (select id from done)
                and active = 1
            ORDER BY {ordering}
            """,
        )

    def delete_previous_coordination(self, old_coordination_id):
        _command(
            self.client,
            f"""
            ALTER TABLE results
            DELETE WHERE eligible = 1
                    and id not in (select id from done)
                    and active = 1
                    and coordination_id = {old_coordination_id}
            """,
            settings=self.alter_settings,
        )

    def insert_mapping(self, mapping_df):
        if not self.does_table_exist("zone_mapping"):
            cols = get_create_table_cols(mapping_df)
            _command(
                self.client,
                f"""
                CREATE TABLE zone_mapping ({",".join(cols)})
                ENGINE = MergeTree()
                ORDER BY (coordination_id, zone_id)
                """,
            )
        _insert_df(self.client, "zone_mapping", mapping_df)

    def get_zone_mapping(self):
        return _query_df(self.client, "select * from zone_mapping")

    def _create_results_table(self, results_df, orderer):
        if self._results_table_exists:
            return
        results_cols = get_create_table_cols(results_df)
        cmd = f"""
        create table if not exists results ({",".join(results_cols)})
            engine = MergeTree() order by ({orderer})
        """
        _command(self.client, cmd, settings=default_insert_settings)

    async def verify(self):
        async def duplicate_tiles():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                "select id from tiles group by id having count(*) > 1",
            )
            if len(df) > 0:
                raise ValueError(f"Duplicate tiles: {df}")

        async def duplicate_results():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                "select id from results group by id having count(*) > 1",
            )
            if len(df) > 0:
                raise ValueError(f"Duplicate results: {df}")

        async def duplicate_done():
            df = _query_df(
                self.client, "select id from done group by id having count(*) > 1"
            )
            if len(df) > 0:
                raise ValueError(f"Duplicate done: {df}")

        async def results_without_tiles():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                """
                select id from results
                    where id not in (select id from tiles)
                """,
            )
            if len(df) > 0:
                raise ValueError(
                    f"Rows in results without corresponding rows in tiles: {df}"
                )

        async def tiles_without_results():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                """
                select id from tiles
                -- packet_id >= 0 excludes tiles that were split or pruned
                    where packet_id >= 0
                        and id not in (select id from results)
                """,
            )
            if len(df) > 0:
                raise ValueError(
                    f"Rows in tiles without corresponding rows in results: {df}"
                )

        async def tiles_without_parents():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                """
                select parent_id, id from tiles
                    where parent_id not in (select id from done)
                """,
            )
            if len(df) > 0:
                raise ValueError(f"tiles without parents: {df}")

        async def tiles_with_active_or_eligible_parents():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                """
                select parent_id, id from tiles
                    where parent_id in (
                        select id from results 
                            where (active=true and id not in 
                                    (select id from done where active=false))
                                or (eligible=true and id not in (select id from done))
                    )
                """,
            )
            if len(df) > 0:
                raise ValueError(f"tiles with active parents: {df}")

        async def inactive_tiles_with_no_children():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                """
                select id from tiles
                -- packet_id >= 0 excludes tiles that were split or pruned
                    where packet_id >= 0
                        and (active=false 
                            or id in (select id from done where active=false))
                        and id not in (select parent_id from tiles)
                """,
            )
            if len(df) > 0:
                raise ValueError(f"inactive tiles with no children: {df}")

        async def refined_tiles_with_incorrect_child_count():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                """
                select d.id, count(*) as n_children, max(refine) as n_expected
                    from done d
                    left join tiles t
                        on t.parent_id = d.id
                    where refine > 0
                    group by d.id
                    having count(*) != max(refine)
                """,
            )
            if len(df) > 0:
                raise ValueError(
                    "refined tiles with wrong number of children:" f" {df}"
                )

        async def deepened_tiles_with_incorrect_child_count():
            df = await asyncio.to_thread(
                _query_df,
                self.client,
                """
                select d.id, count(*) from done d
                    left join tiles t
                        on t.parent_id = d.id
                    where deepen=true
                    group by d.id
                    having count(*) != 1
                """,
            )
            if len(df) > 0:
                raise ValueError(
                    "deepened tiles with wrong number of children:" f" {df}"
                )

        await asyncio.gather(
            duplicate_tiles(),
            duplicate_results(),
            duplicate_done(),
            results_without_tiles(),
            tiles_without_results(),
            tiles_without_parents(),
            tiles_with_active_or_eligible_parents(),
            inactive_tiles_with_no_children(),
            refined_tiles_with_incorrect_child_count(),
            deepened_tiles_with_incorrect_child_count(),
        )

    def create_logs_table(self):
        _command(
            self.client,
            """
            create table if not exists logs (
                worker_id Int32,
                t DateTime64(6),
                name String,
                pathname String,
                lineno UInt32,
                levelno UInt32,
                levelname String,
                message String)
            engine = MergeTree() 
            order by (worker_id, t)
            """,
        )

    def insert_logs(self, df):
        return _insert_df(self.client, "logs", df)

    def close(self):
        self.client.close()

    def get_logs(self):
        return _query_df(self.client, "select * from logs")

    def get_null_hypos(self):
        return _query_df(self.client, "select * from null_hypos")

    def get_config(self):
        return _query_df(self.client, "select * from config")

    def insert_config(self, cfg_df):
        return _insert_df(self.client, "config", cfg_df)

    async def init_grid(self, tiles_df, null_hypos_df, cfg_df):
        # tables:
        # - tiles: id, lots of other stuff...
        # - packet: id
        # - work: id
        # - done: id, active
        # - inactive: materialized view based on done
        tiles_cols = get_create_table_cols(tiles_df)
        cfg_cols = get_create_table_cols(cfg_df)

        id_type = type_map[tiles_df["id"].dtype.name]

        create_null_hypos = asyncio.create_task(
            asyncio.to_thread(
                _command,
                self.client,
                """
                create table null_hypos (
                        description String,
                        serialized String)
                    engine = MergeTree()
                    order by ()
                """,
            )
        )

        create_config = asyncio.create_task(
            asyncio.to_thread(
                _command,
                self.client,
                f"""
                create table config ({','.join(cfg_cols)})
                    engine = MergeTree()
                    order by ()
                """,
            )
        )

        create_done = asyncio.create_task(
            asyncio.to_thread(
                _command,
                self.client,
                f"""
                create table done (
                        zone_id UInt32,
                        step_id UInt32,
                        packet_id Int32,
                        id {id_type},
                        active Bool,
                        finisher_id UInt32,
                        refine UInt32,
                        deepen UInt32,
                        split Bool)
                    engine = MergeTree() 
                    order by (zone_id, step_id, packet_id, id)
                """,
            )
        )

        create_tiles = asyncio.create_task(
            asyncio.to_thread(
                _command,
                self.client,
                f"""
                create table tiles ({",".join(tiles_cols)})
                    engine = MergeTree() 
                    order by (zone_id, step_id, packet_id, id)
                """,
            )
        )

        create_reports = asyncio.create_task(
            asyncio.to_thread(
                _command,
                self.client,
                """
                create table reports (json String)
                    engine = MergeTree() order by ()
                """,
            )
        )

        async def _insert_tiles():
            await create_tiles
            await asyncio.to_thread(self.insert_tiles, tiles_df)

        # Store a reference to the task so that it doesn't get garbage collected.
        insert_tiles_task = asyncio.create_task(_insert_tiles())

        absent_parents = get_absent_parents(tiles_df)

        async def _insert_done():
            await create_done
            await asyncio.to_thread(_insert_df, self.client, "done", absent_parents)

        insert_done_task = asyncio.create_task(_insert_done())

        async def _insert_null_hypos():
            await create_null_hypos
            await asyncio.to_thread(
                _insert_df, self.client, "null_hypos", null_hypos_df
            )

        insert_null_hypos_task = asyncio.create_task(_insert_null_hypos())

        async def _insert_config():
            await create_config
            await asyncio.to_thread(_insert_df, self.client, "config", cfg_df)

        insert_config_task = asyncio.create_task(_insert_config())

        await asyncio.gather(
            create_reports,
            insert_tiles_task,
            insert_done_task,
            insert_null_hypos_task,
            insert_config_task,
        )

    def connect(
        job_id: int = None,
        host=None,
        port=None,
        username=None,
        password=None,
        no_create=False,
    ):
        """
        Connect to a Clickhouse server

        Each job_id corresponds to a Clickhouse database on the Clickhouse
        cluster. If job_id is None, we will find

        For Clickhouse, we will use the following environment variables:
            CLICKHOUSE_HOST: The hostname for the Clickhouse server.
            CLICKHOUSE_PORT: The Clickhouse server port.
            CLICKHOUSE_USERNAME: The Clickhouse username.
            CLICKHOUSE_PASSWORD: The Clickhouse username.

        If the environment variables are not set, the defaults will be:
            port: 8443
            username: "default"

        If the environment variables are not set, the defaults will be:
            port: 37085

        Args:
            job_id: The job_id. Defaults to None.
            host: The hostname for the Clickhouse server. Defaults to None.
            port: The Clickhouse server port. Defaults to None.
            username: The Clickhouse username. Defaults to None.
            password: The Clickhouse password. Defaults to None.
            no_create: If True, do not create the job_id database. Defaults to False.

        Returns:
            A Clickhouse tile database object.
        """
        connection_details = get_ch_config(host, port, username, password)
        if job_id is None:
            test_host = os.environ["CLICKHOUSE_TEST_HOST"]
            if not (
                (test_host is not None and test_host in connection_details["host"])
                or "localhost" in connection_details["host"]
            ):
                raise RuntimeError(
                    "To run a production job, please choose an explicit unique job_id."
                )
            job_id = "unnamed_" + uuid.uuid4().hex

        client = get_ch_client(connection_details=connection_details)
        if not no_create:
            # Create job_id database if it doesn't exist
            _command(
                client,
                f"create database if not exists {job_id}",
                settings=default_insert_settings,
            )

        # NOTE: client.database is invading private API, but based on reading
        # the clickhouse_connect code, this is unlikely to break
        client.database = job_id

        logger.info(f"Connected to job {job_id}")
        return Clickhouse(connection_details, client, job_id)


def get_ch_client(
    connection_details=None,
    host=None,
    port=None,
    username=None,
    password=None,
    database=None,
):
    import clickhouse_connect

    clickhouse_connect.common.set_setting("autogenerate_session_id", False)
    if connection_details is None:
        connection_details = get_ch_config(host, port, username, password, database)
    pool_mgr = clickhouse_connect.driver.httputil.get_pool_manager(maxsize=16)
    return clickhouse_connect.get_client(**connection_details, pool_mgr=pool_mgr)


def get_ch_config(host=None, port=None, username=None, password=None, database=None):
    if host is None:
        if "CLICKHOUSE_HOST" in os.environ:
            host = os.environ["CLICKHOUSE_HOST"]
        else:
            host = os.environ["CLICKHOUSE_TEST_HOST"]
    if port is None:
        if "CLICKHOUSE_PORT" in os.environ:
            port = os.environ["CLICKHOUSE_PORT"]
        else:
            port = 8443
    if username is None:
        if "CLICKHOUSE_USERNAME" in os.environ:
            username = os.environ["CLICKHOUSE_USERNAME"]
        else:
            username = "default"
    if password is None:
        password = os.environ["CLICKHOUSE_PASSWORD"]
    logger.info(f"Clickhouse config: {username}@{host}:{port}/{database}")
    return dict(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
    )


def clear_dbs(ch_client=None, names=None, yes=False):
    """
    DANGER, WARNING, ACHTUNG, PELIGRO:
        Don't run this function for our production Clickhouse server. That
        would be bad. There's a built-in safeguard to prevent this, but it's
        not foolproof.

    Clear all databases (and database tables) from the Clickhouse. This should
    only work for our test database or for localhost.

    Args:
        ch_client: Clickhouse client
        names: default None, list of database names to drop. If None, drop all.
        yes: bool, if True, don't ask for confirmation before dropping.
    """
    if ch_client is None:
        ch_client = get_ch_client()

    test_host = os.environ["CLICKHOUSE_TEST_HOST"]
    if not (
        (test_host is not None and test_host in ch_client.url)
        or "localhost" in ch_client.url
    ):
        raise RuntimeError("This function is only for localhost or test databases.")

    if names is None:
        to_drop = []
        all_dbs = retry_ch_action(ch_client.query_df, "show databases")
        for db in all_dbs["name"]:
            if db not in [
                "default",
                "INFORMATION_SCHEMA",
                "information_schema",
                "system",
            ]:
                to_drop.append(db)
    else:
        to_drop = names

    if len(to_drop) == 0:
        print("No Clickhouse databases to drop.")
    else:
        print("Dropping the following databases:")
        print(to_drop)
        if not yes:
            print("Are you sure? [yN]", flush=True)
            yes = input() == "y"

        if yes:
            for db in to_drop:
                cmd = f"drop database {db}"
                print(cmd)
                _command(ch_client, cmd)
