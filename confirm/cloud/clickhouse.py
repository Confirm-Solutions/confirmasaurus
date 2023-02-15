"""
Some notes I don't want to lose:

Speeding up clickhouse stuff by using threading and specifying column_types:
    https://clickhousedb.slack.com/archives/CU478UEQZ/p1673560678632889

Fast min/max with clickhouse:
    https://clickhousedb.slack.com/archives/CU478UEQZ/p1669820710989079
"""
import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Dict
from typing import List

import clickhouse_connect
import pandas as pd
import pyarrow
import redis

import confirm.adagrid.json as json
import imprint.log
from .redis_heartbeat import HeartbeatThread
from confirm.adagrid.store import is_table_name
from confirm.adagrid.store import Store

clickhouse_connect.common.set_setting("autogenerate_session_id", False)

logger = imprint.log.getLogger(__name__)

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
}


def get_create_table_cols(df):
    """
    Map from Pandas dtypes to Clickhouse types.

    Args:
        df: The dataframe

    Returns:
        A list of strings of the form "column_name type"
    """
    return [f"{c} {type_map[dt.name]}" for c, dt in zip(df.columns, df.dtypes)]


def _query_df(client, query):
    # Loading via Arrow and then converting to Pandas is faster than using
    # query_df directly to load a pandas dataframe. I'm guessing that loading
    # through arrow is a little less flexible or something, but for our
    # purposes, this is great.
    return client.query_arrow(query).to_pandas()


def _insert_df(client, table, df):
    # Save as _query_df, inserting through arrow is faster!
    client.insert_arrow(table, pyarrow.Table.from_pandas(df, preserve_index=False))


@dataclass
class ClickhouseStore(Store):
    """
    A store using Clickhouse as the backend.
    """

    client: clickhouse_connect.driver.httpclient.HttpClient
    job_id: str

    def __post_init__(self):
        self.client.command(
            """
            create table if not exists store_tables
                (key String, table_name String)
                order by key
            """
        )

    def get(self, key):
        exists, table_name = self._exists(key)
        logger.debug(f"get({key}) -> {exists} {table_name}")
        if exists:
            return _query_df(self.client, f"select * from {table_name}")
        else:
            raise KeyError(f"Key {key} not found in store")

    def set(self, key, df, nickname=None):
        exists, table_name = self._exists(key)
        if exists:
            self.client.command(f"drop table {table_name}")
        else:
            table_id = uuid.uuid4().hex
            if is_table_name(key):
                table_name = key
            else:
                table_name = (
                    f"_store_{nickname}_{table_id}"
                    if nickname is not None
                    else f"_store_{table_id}"
                )
            self.client.command(
                "insert into store_tables values (%s, %s)",
                (key, table_name),
            )
        logger.debug(f"set({key}) -> {exists} {table_name}")
        cols = get_create_table_cols(df)
        self.client.command(
            f"""
            create table {table_name} ({",".join(cols)})
                engine = MergeTree() order by tuple()
        """
        )
        _insert_df(self.client, self.job_id + "." + table_name, df)

    def set_or_append(self, key, df):
        exists, table_name = self._exists(key)
        logger.debug(f"set_or_append({key}) -> {exists} {table_name}")
        if exists:
            _insert_df(self.client, table_name, df)
        else:
            self.set(key, df)

    def _exists(self, key):
        table_name = self.client.query(
            "select table_name from store_tables where key = %s", (key,)
        ).result_set
        if len(table_name) == 0:
            return False, None
        else:
            return True, table_name[0][0]

    def exists(self, key):
        out = self._exists(key)[0]
        logger.debug(f"exists({key}) = {out}")
        return out


@dataclass
class Clickhouse:
    """
    A tile database built on top of Clickhouse. This should be very fast and
    robust and is preferred for large runs. Latency will be worse than with
    DuckDB because network requests will be required for each query.

    See the DuckDBTiles or PandasTiles implementations for details on the
    Adagrid tile database interface.

    Internally, this also uses a Redis server for distributed locks and for
    robustly incrementing the worker_id. We don't use Clickhouse for these
    tasks because Clickhouse does not have any strong consistency tools or
    tools for locking a table.

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
    client: clickhouse_connect.driver.httpclient.HttpClient
    redis_con: redis.Redis
    job_id: str
    _tiles_columns_cache: List[str] = None
    _results_columns_cache: List[str] = None
    _d: int = None
    _results_table_exists: bool = False
    is_distributed: bool = True
    supports_threads: bool = True

    def __post_init__(self):
        self.lock = redis.lock.Lock(
            self.redis_con, f"{self.job_id}:next_lock", timeout=60, blocking_timeout=3
        )

    @property
    def _host(self):
        return self.connection_details["host"]

    @property
    def store(self):
        return ClickhouseStore(self.client, self.job_id)

    def heartbeat(self, worker_id):
        return HeartbeatThread(self.redis_con, self.job_id, worker_id)

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

    def get_tiles(self):
        cols = ",".join(
            [c for c in self._tiles_columns() if c not in ["active", "eligible"]]
        )
        return _query_df(
            self.client,
            f"""
            select {cols},
                and(active=true,
                    (id not in (select id from inactive)))
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
                    (id not in (select id from inactive)))
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

    def get_reports(self):
        json_strs = self.client.query("select * from reports").result_set
        return pd.DataFrame([json.loads(s[0]) for s in json_strs]).sort_values(
            by=["worker_id", "step_id", "packet_id"]
        )

    def insert_report(self, report):
        self.client.command(f"insert into reports values ('{json.dumps(report)}')")

    def n_processed_tiles(self, worker_id: int, step_id: int) -> int:
        # This checks the number of tiles for which both:
        # - there are results
        # - the parent tile is done
        # This is a good way to check if a step is done because it ensures that
        # all relevant inserts are done.
        #
        if not self._results_table_exists:
            self._results_table_exists = does_table_exist(
                self.client, self.job_id, "results"
            )
            if not self._results_table_exists:
                return -1

        R = self.client.query(
            f"""
            select count(*) from tiles
                where
                    worker_id = {worker_id}
                    and step_id = {step_id}
                    and active = true
                    and id in (select id from results)
                    and parent_id in (select id from done)
            """
        )
        return R.result_set[0][0]

    def get_coordination_id(self):
        return int(
            self.db.redis_con.get(f"{self.db.job_id}:coordination_id").decode("ascii")
        )

    def get_starting_step_id(self, worker_id):
        prefix = f"{self.job_id}:worker_{worker_id}:step_"
        suffix = ":n_tiles"
        started_steps = list(
            [
                int(k[len(prefix) : -len(suffix)])
                for k in self.redis_con.scan_iter(f"{prefix}*{suffix}")
            ]
        )
        return max(started_steps)

    def set_step_info(self, worker_id, new_step_id, n_tiles, n_packets):
        p = self.redis_con.pipeline()
        p.set(
            f"{self.job_id}:worker_{worker_id}:step_{new_step_id}:n_tiles",
            n_tiles,
        )
        p.set(
            f"{self.job_id}:worker_{worker_id}:step_{new_step_id}:n_packets", n_packets
        )
        p.execute()

    def get_step_info(self, worker_id, step_id):
        p = self.redis_con.pipeline()
        p.get(f"{self.job_id}:worker_{worker_id}:step_{step_id}:n_tiles")
        p.get(f"{self.job_id}:worker_{worker_id}:step_{step_id}:n_packets")
        n_tiles, n_packets = p.execute()
        return int(n_tiles), int(n_packets)

    def insert_tiles(self, df: pd.DataFrame):
        logger.debug(f"writing {df.shape[0]} tiles")
        _insert_df(self.client, "tiles", df)

    def insert_results(self, df: pd.DataFrame, orderer: str):
        self._create_results_table(df, orderer)
        logger.debug(f"writing {df.shape[0]} results")
        _insert_df(self.client, "results", df)
        self._results_table_exists = True

    def finish(self, which):
        logger.debug(f"finish: {which.head()}")
        _insert_df(self.client, "done", which)

    def worst_tile(self, worker_id, orderer):
        if worker_id is None:
            worker_id_clause = ""
        else:
            worker_id_clause = f"and worker_id = {worker_id}"

        return _query_df(
            self.client,
            f"""
            select * from results
                where
                    active=true
                    and id not in (select id from inactive)
                    {worker_id_clause}
            order by {orderer} limit 1
        """,
        )

    def get_packet(self, coordination_id, worker_id, step_id, packet_id):
        """
        `get_packet` is used to select tiles for processing/simulation.
        """
        return _query_df(
            self.client,
            f"""
            select * from tiles
                where
                    coordination_id = {coordination_id}
                    and worker_id = {worker_id}
                    and step_id = {step_id}
                    and packet_id = {packet_id}
            """,
        )

    def check_packet_flag(self, worker_id, step_id, packet_id):
        flag_name = f"{self.job_id}:worker_{worker_id}:step_{step_id}:packet_{packet_id}:insert"  # noqa
        flag_val = self.redis_con.get(flag_name)
        if flag_val is None:
            return None
        else:
            return int(flag_val.decode("ascii"))

    def set_packet_flag(self, worker_id, step_id, packet_id):
        flag_name = f"{self.job_id}:worker_{worker_id}:step_{step_id}:packet_{packet_id}:insert"  # noqa
        return self.redis_con.setnx(flag_name, worker_id) == 1

    def next(self, coordination_id, worker_id, n, orderer):
        """
        `next` is used in select_tiles to get the next batch of tiles to
        refine/deepen
        """
        return _query_df(
            self.client,
            f"""
            select * from results
                where
                    coordination_id = {coordination_id}
                    and worker_id = {worker_id}
                    and eligible=true
                    and id not in (select id from done)
            order by {orderer} limit {n}
        """,
        )

    def bootstrap_lamss(self, worker_id):
        # Get the number of bootstrap lambda* columns
        nB = (
            max([int(c[6:]) for c in self._results_columns() if c.startswith("B_lams")])
            + 1
        )

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        if worker_id is None:
            worker_id_clause = ""
        else:
            worker_id_clause = f"and worker_id = {worker_id}"
        lamss = self.client.query(
            f"""
            select {cols} from results where
                active=true
                and id not in (select id from inactive)
                {worker_id_clause}
        """
        ).result_set[0]

        return lamss

    def _create_results_table(self, results_df, orderer):
        if self._results_table_exists:
            return
        results_cols = get_create_table_cols(results_df)
        cmd = f"""
        create table if not exists results ({",".join(results_cols)})
            engine = MergeTree() order by (coordination_id, {orderer})
        """
        self.client.command(cmd)

    def close(self):
        self.client.close()
        self.redis_con.close()

    def new_workers(self, n):
        with self.lock:
            p = self.redis_con.pipeline()
            p.setnx(f"{self.job_id}:next_worker_id", 2)
            p.incrby(f"{self.job_id}:next_worker_id", n)
            _, cur_id = p.execute()
        return [cur_id - n + i for i in range(n)]

    async def init_tiles(self, tiles_df, wait=False):
        # tables:
        # - tiles: id, lots of other stuff...
        # - packet: id
        # - work: id
        # - done: id, active
        # - inactive: materialized view based on done
        tiles_cols = get_create_table_cols(tiles_df)

        id_type = type_map[tiles_df["id"].dtype.name]

        create_done = asyncio.create_task(
            asyncio.to_thread(
                self.client.command,
                f"""
                create table done (
                        coordination_id UInt32,
                        worker_id UInt32,
                        step_id UInt32,
                        packet_id Int32,
                        id {id_type},
                        active Bool,
                        finisher_id UInt32,
                        refine Bool,
                        deepen Bool,
                        split Bool)
                    engine = MergeTree() 
                    order by (coordination_id, worker_id, step_id, packet_id, id)
                """,
            )
        )

        create_tiles = asyncio.create_task(
            asyncio.to_thread(
                self.client.command,
                f"""
            create table tiles ({",".join(tiles_cols)})
                engine = MergeTree() 
                order by (coordination_id, worker_id, step_id, packet_id, id)
            """,
            )
        )

        create_reports = asyncio.create_task(
            asyncio.to_thread(
                self.client.command,
                """
            create table reports (json String)
                engine = MergeTree() order by ()
            """,
            )
        )

        # these tiles have no parents. poor sad tiles :(
        # we need to put these absent parents into the done table so that
        # n_processed_tiles returns correct results.
        absent_parents = pd.DataFrame(
            tiles_df["parent_id"].unique()[:, None], columns=["id"]
        )
        for c in [
            "worker_id",
            "step_id",
            "packet_id",
            "active",
            "finisher_id",
            "refine",
            "deepen",
            "split",
        ]:
            absent_parents[c] = 0

        async def _create_inactive():
            await create_done
            await asyncio.to_thread(
                self.client.command,
                """
            create materialized view inactive
                engine = MergeTree()
                order by (coordination_id, worker_id, step_id, id)
                as select coordination_id, worker_id, step_id, id from done 
                where active=false
            """,
            )

        create_inactive = asyncio.create_task(_create_inactive())

        async def _insert_tiles():
            await create_tiles
            await asyncio.to_thread(self.insert_tiles, tiles_df)

        # Store a reference to the task so that it doesn't get garbage collected.
        self._insert_tiles_task = asyncio.create_task(_insert_tiles())

        async def _insert_done():
            await create_done
            await asyncio.to_thread(_insert_df, self.client, "done", absent_parents)

        self._insert_done_task = asyncio.create_task(_insert_done())
        set_coordination_id = asyncio.create_task(
            asyncio.to_thread(self.redis_con.set, f"{self.job_id}:coordination_id", 0)
        )

        def _setup_first_step():
            p = self.redis_con.pipeline()
            for w, w_n_tiles in tiles_df.groupby("worker_id").size().items():
                p.set(f"{self.job_id}:worker_{w}:step_0:n_tiles", w_n_tiles)
            for w, max_packet_id in (
                tiles_df.groupby("worker_id")["packet_id"].max().items()
            ):
                p.set(f"{self.job_id}:worker_{w}:step_0:n_packets", max_packet_id + 1)
            p.execute()

        setup_first_step = asyncio.create_task(asyncio.to_thread(_setup_first_step))

        # I think the only things that we *need* to wait for are the create
        # table statements. The inserts are not urgent.
        wait_for = [
            create_reports,
            create_tiles,
            create_done,
            create_inactive,
            set_coordination_id,
            setup_first_step,
        ]
        if wait:
            wait_for.extend([self._insert_tiles_task, self._insert_done_task])
        await asyncio.gather(*wait_for)

    def connect(
        job_id: int = None,
        host=None,
        port=None,
        username=None,
        password=None,
        redis_host=None,
        redis_port=None,
        redis_password=None,
    ):
        """
        Connect to a Clickhouse server and to our Redis server.

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

        For Redis, we will use the following environment variables:
            REDIS_HOST: The hostname for the Redis server.
            REDIS_PORT: The Redis server port.
            REDIS_PASSWORD: The Redis password.

        If the environment variables are not set, the defaults will be:
            port: 37085

        Args:
            job_id: The job_id. Defaults to None.
            host: The hostname for the Clickhouse server. Defaults to None.
            port: The Clickhouse server port. Defaults to None.
            username: The Clickhouse username. Defaults to None.
            password: The Clickhouse password. Defaults to None.
            redis_host: The Redis hostname. Defaults to None.
            redis_port: The Redis port. Defaults to None.
            redis_password: The Redis password. Defaults to None.

        Returns:
            A Clickhouse tile database object.
        """
        config = get_ch_config(host, port, username, password)
        if job_id is None:
            test_host = os.environ["CLICKHOUSE_TEST_HOST"]
            if not (
                (test_host is not None and test_host in config["host"])
                or "localhost" in config["host"]
            ):
                raise RuntimeError(
                    "To run a production job, please choose an explicit unique job_id."
                )
            job_id = uuid.uuid4().hex

        # Create job_id database if it doesn't exist
        client = clickhouse_connect.get_client(**config)
        client.command(f"create database if not exists {job_id}")

        connection_details = get_ch_config(host, port, username, password, job_id)
        client = clickhouse_connect.get_client(**connection_details)

        redis_con = get_redis_client(redis_host, redis_port, redis_password)
        logger.info(f"Connected to job {job_id}")
        return Clickhouse(connection_details, client, redis_con, job_id)


def does_table_exist(client, job_id: str, table_name: str) -> bool:
    return (
        len(
            client.query(
                f"""
        select * from information_schema.tables
            where table_schema = '{job_id}'
                and table_name = '{table_name}'
        """
            ).result_set
        )
        > 0
    )


def get_redis_client(host=None, port=None, password=None):
    return redis.Redis(**get_redis_config(host, port, password))


def get_redis_config(host=None, port=None, password=None):
    if host is None:
        host = os.environ["REDIS_HOST"]
    if port is None:
        if "REDIS_PORT" in os.environ:
            port = os.environ["REDIS_PORT"]
        else:
            port = 37085
    if password is None:
        password = os.environ["REDIS_PASSWORD"]
    return dict(host=host, port=port, password=password)


def get_ch_client(host=None, port=None, username=None, password=None, job_id=None):
    connection_details = get_ch_config(host, port, username, password, job_id)
    return clickhouse_connect.get_client(**connection_details)


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
        host=host, port=port, username=username, password=password, database=database
    )


def clear_dbs(
    ch_client=None, redis_client=None, names=None, yes=False, drop_all_redis_keys=False
):
    """
    DANGER, WARNING, ACHTUNG, PELIGRO:
        Don't run this function for our production Clickhouse server. That
        would be bad. There's a built-in safeguard to prevent this, but it's
        not foolproof.

    Clear all databases (and database tables) from the Clickhouse and Redis
    servers. This should only work for our test database or for localhost.

    Args:
        ch_client: Clickhouse client
        redis_client: Redis client
        names: default None, list of database names to drop. If None, drop all.
        yes: bool, if True, don't ask for confirmation before dropping.
        drop_all_redis_keys: bool, if True, drop all Redis keys.
    """
    if ch_client is None:
        ch_client = get_ch_client()
    if redis_client is None:
        redis_client = get_redis_client()

    test_host = os.environ["CLICKHOUSE_TEST_HOST"]
    if not (
        (test_host is not None and test_host in ch_client.url)
        or "localhost" in ch_client.url
    ):
        raise RuntimeError("This function is only for localhost or test databases.")

    if names is None:
        to_drop = []
        for db in ch_client.query_df("show databases")["name"]:
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
                ch_client.command(cmd)
                if redis_client is not None:
                    keys = list(redis_client.scan_iter(f"*{db}*")) + list(
                        redis_client.scan_iter(f"{db}*")
                    )
                    print(f"deleting redis keys containing '{db}'")
                    redis_client.delete(*keys)
                else:
                    print("no redis client, skipping redis keys")

    if drop_all_redis_keys:
        keys = list(redis_client.scan_iter("*"))
        print("drop_all_redis_keys=True, deleting ALL redis keys!")
        if not yes:
            print("Are you sure? [yN]", flush=True)
            yes = input() == "y"
        redis_client.delete(*keys)
