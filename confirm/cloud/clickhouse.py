"""
Some notes I don't want to lose:

Speeding up clickhouse stuff by using threading and specifying column_types:
    https://clickhousedb.slack.com/archives/CU478UEQZ/p1673560678632889

Fast min/max with clickhouse:
    https://clickhousedb.slack.com/archives/CU478UEQZ/p1669820710989079
"""
import uuid
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict
from typing import List

import clickhouse_connect
import dotenv
import pandas as pd
import pyarrow
import redis

import imprint.log
from confirm.adagrid.store import is_table_name
from confirm.adagrid.store import Store

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
    _tiles_columns: List[str] = None
    _results_columns: List[str] = None
    _d: int = None
    _results_table_exists: bool = False

    def __post_init__(self):
        self.lock = redis.lock.Lock(self.redis_con, f"{self.job_id}:next_lock", timeout=60)

    @property
    def _host(self):
        return self.connection_details["host"]

    @property
    def store(self):
        return ClickhouseStore(self.client, self.job_id)

    def dimension(self):
        if self._d is None:
            cols = self.tiles_columns()
            self._d = max([int(c[5:]) for c in cols if c.startswith("theta")]) + 1
        return self._d

    def tiles_columns(self):
        if self._tiles_columns is None:
            self._tiles_columns = _query_df(
                self.client, "select * from tiles limit 1"
            ).columns
        return self._tiles_columns

    def results_columns(self):
        if self._results_columns is None:
            self._results_columns = _query_df(
                self.client, "select * from results limit 1"
            ).columns
        return self._results_columns

    def get_tiles(self):
        cols = ",".join(
            [c for c in self.tiles_columns() if c not in ["active", "eligible"]]
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
            [c for c in self.results_columns() if c not in ["active", "eligible"]]
        )
        return _query_df(
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

    def get_step_info(self):
        out = literal_eval(self.redis_con.get(f"{self.job_id}:step_info").decode())
        logger.debug("get step info: %s", out)
        return out

    def set_step_info(self, *, step_id: int, step_iter: int, n_iter: int, n_tiles: int):
        step_info = (step_id, step_iter, n_iter, n_tiles)
        logger.debug("set step info: %s", step_info)
        self.redis_con.set(f"{self.job_id}:step_info", str(step_info))

    def n_processed_tiles(self, step_id: int) -> int:
        # This checks the number of tiles for which both:
        # - there are results
        # - the parent tile is done
        # This is a good way to check if a step is done because it ensures that
        # all relevant inserts are done.
        #
        if not self._results_table_exists:
            self._results_table_exists = does_table_exist(self.client, "results")
            if not self._results_table_exists:
                return -1

        R = self.client.query(
            f"""
            select count(*) from tiles
                where
                    step_id = {step_id}
                    and id in (select id from results)
                    and parent_id in (select id from done)
            """
        )
        return R.result_set[0][0]

    def insert_tiles(self, df: pd.DataFrame):
        logger.debug(f"writing {df.shape[0]} tiles")
        _insert_df(self.client, "tiles", df)

    def insert_results(self, df: pd.DataFrame):
        self._create_results_table(df)
        logger.debug(f"writing {df.shape[0]} results")
        _insert_df(self.client, "results", df)

    def worst_tile(self, order_col):
        return _query_df(
            self.client,
            f"""
            select * from results r
                where
                    active=true
                    and id not in (select id from inactive)
            order by {order_col} limit 1
        """,
        )

    def get_work(self, step_id, step_iter):
        return _query_df(
            self.client,
            f"""
            select * from tiles
                where
                    step_id = {step_id}
                    and step_iter = {step_iter}
            """,
        )

    def select_tiles(self, n, order_col):
        return _query_df(
            self.client,
            f"""
            select * from results
                where
                    eligible=true
                    and id not in (select id from done)
            order by {order_col} asc limit {n}
        """,
        )

    def finish(self, which):
        logger.debug(f"finish: {which.head()}")
        _insert_df(self.client, "done", which)

    def bootstrap_lamss(self):
        # Get the number of bootstrap lambda* columns
        nB = (
            max([int(c[6:]) for c in self.results_columns() if c.startswith("B_lams")])
            + 1
        )

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        lamss = self.client.query(
            f"""
            select {cols} from results where
                active=true
                and id not in (select id from inactive)
        """
        ).result_set[0]

        return lamss

    def _create_results_table(self, results_df):
        if self._results_table_exists:
            return
        self._results_table_exists = True
        results_cols = get_create_table_cols(results_df)
        cmd = f"""
        create table if not exists results ({",".join(results_cols)})
            engine = MergeTree() order by orderer
        """
        self.client.command(cmd)
        self.insert_results(results_df)

    def close(self):
        self.client.close()
        self.redis_con.close()

    def init_tiles(self, tiles_df):
        # tables:
        # - tiles: id, lots of other stuff...
        # - packet: id
        # - work: id
        # - done: id, active
        # - inactive: materialized view based on done
        tiles_cols = get_create_table_cols(tiles_df)

        id_type = type_map[tiles_df["id"].dtype.name]
        commands = [
            f"""
            create table tiles ({",".join(tiles_cols)})
                engine = MergeTree() order by (step_id, step_iter, id)
            """,
            # TODO: could leave this uncreated until we need it
            f"""
            create table done (
                    id {id_type},
                    step_id UInt32,
                    step_iter UInt32,
                    active Bool,
                    query_time Float64,
                    finisher_id UInt32,
                    refine Bool,
                    deepen Bool)
                engine = MergeTree() order by id
            """,
            """
            create materialized view inactive
                engine = MergeTree() order by id
                as select id from done where active=false
            """,
        ]
        for c in commands:
            self.client.command(c)
        self.client.command("insert into done values (0, 0, 0, 0, 0, 0, 0, 0)")
        self.insert_tiles(tiles_df)
        self.set_step_info(step_id=-1, step_iter=0, n_iter=0, n_tiles=0)

    def new_worker(self):
        with self.lock:
            return self.redis_con.incr(f"{self.job_id}:worker_id") + 1

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
            test_host = dotenv.dotenv_values()["CLICKHOUSE_TEST_HOST"]
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


def does_table_exist(client, table_name: str) -> bool:
    return (
        len(
            client.query(
                f"""
        select * from information_schema.schemata
            where schema_name = '{table_name}'
        """
            ).result_set
        )
        > 0
    )


def get_redis_client(host=None, port=None, password=None):
    return redis.Redis(**get_redis_config(host, port, password))


def get_redis_config(host=None, port=None, password=None):
    env = dotenv.dotenv_values()
    if host is None:
        host = env["REDIS_HOST"]
    if port is None:
        if "REDIS_PORT" in env:
            port = env["REDIS_PORT"]
        else:
            port = 37085
    if password is None:
        password = env["REDIS_PASSWORD"]
    return dict(host=host, port=port, password=password)


def get_ch_client(host=None, port=None, username=None, password=None, job_id=None):
    connection_details = get_ch_config(host, port, username, password, job_id)
    return clickhouse_connect.get_client(**connection_details)


def get_ch_config(host=None, port=None, username=None, password=None, database=None):
    env = dotenv.dotenv_values()
    if host is None:
        if "CLICKHOUSE_HOST" in env:
            host = env["CLICKHOUSE_HOST"]
        else:
            host = env["CLICKHOUSE_TEST_HOST"]
    if port is None:
        if "CLICKHOUSE_PORT" in env:
            port = env["CLICKHOUSE_PORT"]
        else:
            port = 8443
    if username is None:
        if "CLICKHOUSE_USERNAME" in env:
            username = env["CLICKHOUSE_USERNAME"]
        else:
            username = "default"
    if password is None:
        password = env["CLICKHOUSE_PASSWORD"]
    logger.info(f"Clickhouse config: {username}@{host}:{port}/{database}")
    return dict(
        host=host, port=port, username=username, password=password, database=database
    )


def clear_dbs(ch_client, redis_client, names=None, yes=False, drop_all_redis_keys=False):
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
    test_host = dotenv.dotenv_values()["CLICKHOUSE_TEST_HOST"]
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
            print("Are you sure? [yN]")
            yes = input() == "y"

        if yes:
            for db in to_drop:
                cmd = f"drop database {db}"
                print(cmd)
                ch_client.command(cmd)
                if redis_client is not None:
                    keys = list(redis_client.scan_iter("*{db}*"))
                    print("deleting redis keys", keys)
                    redis_client.delete(*keys)
                else:
                    print("no redis client, skipping redis keys")
    
    if drop_all_redis_keys:
        keys = list(redis_client.scan_iter("*"))
        print("drop_all_redis_keys=True, deleting redis keys", keys)
        if not yes:
            print("Are you sure? [yN]")
            yes = input() == "y"
        redis_client.delete(*keys)
