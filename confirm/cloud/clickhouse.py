import os
import time
import uuid
from dataclasses import dataclass
from typing import List

import clickhouse_connect
import pottery

try:
    import keyring

    assert keyring.get_keyring().priority
except (ImportError, AssertionError):
    # No suitable keyring is available, so mock the interface
    # to simulate no pw.
    # https://github.com/jeffwidman/bitbucket-issue-migration/commit/f4a2e18b1a8e54ee8e265bf71d0808c5a99f66f9
    class keyring:
        get_password = staticmethod(lambda system, username: None)


import pyarrow
import redis

from confirm.adagrid.store import is_table_name
from confirm.adagrid.store import Store

type_map = {
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
        print(exists, table_name)
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
                "insert into store_tables values (%s, %s)", (key, table_name)
            )
        cols = get_create_table_cols(df)
        # if "id" not in df.columns:
        #     df["id"] = df.index
        self.client.command(
            f"""
            create table {table_name} ({",".join(cols)})
                engine = MergeTree() order by tuple()
        """
        )
        _insert_df(self.client, table_name, df)

    def set_or_append(self, key, df):
        exists, table_name = self._exists(key)
        if exists:
            print("exists", table_name, df)
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
        return self._exists(key)[0]


@dataclass
class Clickhouse:
    """
    A tile database built on top of Clickhouse. This should be very fast and
    robust and is preferred for large runs. Latency will be worse than with
    DuckDB because a network request will be required for each query.

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
        - the "tiles_inactive" table, which contains tiles that are inactive
        - a tile is only active is active=True and id not in tiles_inactive

    eligible: True if the tile is eligible to be selected for work. Specified in
         two ways
        - the "eligibile" column in the tiles table
        - the "work" table, which contains tiles that are ineligible because
          they either are being worked on or have been worked on
        - a tile is only eligible if eligible=True and id not in tiles_ineligible

    The reason for these dual sources of information is the desire to only
    rarely update rows that have already been written. Updating data is painful
    in Clickhouse. We could periodically flush the tiles_ineligible and
    tiles_inactive tables to corresponding columns in the tiles table. This
    would reduce the size of the subqueries and probably improve performance
    for very large jobs. But, this flushing would probably only need to happen
    once every few hours.

    NOTE: A tile may be active but ineligible if it was selected for work but the
    criterion decided it was not worth refining or deepening.

    NOTE: We could explore a Clickhouse projection or a Clickhouse materialized
    view as alternative approaches to the dual sources of information for
    active and eligible tiles.
    """

    client: clickhouse_connect.driver.httpclient.HttpClient
    redis_con: redis.Redis
    host: str
    job_id: str
    worker_id: int
    _columns: List[str] = None
    _d: int = None

    @property
    def store(self):
        return ClickhouseStore(self.client)

    def dimension(self):
        if self._d is None:
            self._d = (
                max([int(c[5:]) for c in self.columns() if c.startswith("theta")]) + 1
            )
        return self._d

    def columns(self):
        if self._columns is None:
            self._columns = _query_df(
                self.client, "select * from tiles limit 1"
            ).columns
        return self._columns

    def get_all(self):
        cols = ",".join([c for c in self.columns() if c not in ["active", "eligible"]])
        return _query_df(
            self.client,
            f"""
            select {cols},
                and(active=true, (id not in (select id from tiles_inactive))) as active,
                and(eligible=true, (id not in (select id from work))) as eligible
            from tiles
        """,
        )

    def write(self, df):
        _insert_df(self.client, "tiles", df)

    def worst_tile(self, order_col):
        return _query_df(
            self.client,
            f"""
            select * from tiles where
                active=true
                and id not in (select * from tiles_inactive)
            order by {order_col} limit 1
        """,
        )

    def next(self, n, order_col):
        lock = pottery.Redlock(
            key=f"{self.host}/{self.job_id}/lock", masters={self.redis_con}
        )
        with lock:
            out = _query_df(
                self.client,
                f"""
                select * from tiles where
                    eligible=true
                    and id not in (select id from work)
                order by {order_col} asc limit {n}
            """,
            )
            T = time.time()
            out["time"] = T
            out["worker_id"] = self.worker_id
            _insert_df(self.client, "work", out[["id", "time", "worker_id"]])

        # No need to check for conflicts because we are using a lock but if we
        # were to check for conflicts, it would look something like this
        # conflicts = _query_df(
        # self.client,
        #     f"""
        #     with mine as (
        #         select id, time from work where
        #             worker_id={self.worker_id}
        #             and time={T}
        #     )
        #     select worker_id, time from work
        #         join mine on (
        #             work.id=mine.id
        #             and work.worker_id != {self.worker_id}
        #         )
        #         where work.time <= mine.time
        # """
        # )
        # assert conflicts.shape[0] == 0
        return out

    def finish(self, which):
        _insert_df(self.client, "tiles_inactive", which.loc[~which["active"]][["id"]])

    def bootstrap_lamss(self):
        # Get the number of bootstrap lambda* columns
        nB = max([int(c[6:]) for c in self.columns() if c.startswith("B_lams")]) + 1

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        lamss = self.client.query(
            f"""
            select {cols} from tiles where
                active=true
                and id not in (select id from tiles_inactive)
        """
        ).result_set[0]

        return lamss

    def close(self):
        self.client.close()

    def init_tiles(self, df):
        cols = get_create_table_cols(df)
        orderby = "orderer" if "orderer" in df.columns else "id"
        self.client.command(
            f"""
            create table tiles ({",".join(cols)})
                engine = MergeTree() order by {orderby}
        """
        )

        id_type = type_map[df["id"].dtype.name]
        self.client.command(
            f"""
            create table tiles_inactive (id {id_type})
                engine = MergeTree() order by id
            """
        )
        self.client.command(
            f"""
            create table work
                (id {id_type}, time Float64, worker_id UInt32)
                engine = MergeTree() order by (worker_id, id)
            """
        )
        self.write(df)

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
            host: keyring entry "clickhouse-confirm-test-host"
            port: 8443
            username: "default"
            password: keyring entry "clickhouse-confirm-test-password"

        For Redis, we will use the following environment variables:
            REDIS_HOST: The hostname for the Redis server.
            REDIS_PORT: The Redis server port.
            REDIS_PASSWORD: The Redis password.

        If the environment variables are not set, the defaults will be:
            host: keyring entry "upstash-confirm-coordinator-host"
            port: 37085
            password: keyring entry "upstash-confirm-coordinator-password"

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
        if job_id is None:
            config = get_ch_config(host, port, username, password)
            test_host = keyring.get_password(
                "clickhouse-confirm-test-host", os.environ["USER"]
            )
            if not (
                (test_host is not None and test_host in config["host"])
                or "localhost" in config["host"]
            ):
                raise RuntimeError(
                    "To run a production job, please choose an explicit unique job_id."
                )
            client = clickhouse_connect.get_client(**config)
            job_id = find_unique_job_id(client)
        connection_details = get_ch_config(host, port, username, password, job_id)
        host = connection_details["host"]
        client = clickhouse_connect.get_client(**connection_details)
        redis_con = redis.Redis(
            **get_redis_config(redis_host, redis_port, redis_password)
        )
        worker_id = next(
            pottery.NextId(
                key=f"{host}/{job_id}/worker_id",
                masters={redis_con},
            )
        )
        print(f"Connected to job {job_id} as worker {worker_id}")
        return Clickhouse(client, redis_con, host, job_id, worker_id)


def get_redis_config(host=None, port=None, password=None):
    if host is None:
        if "REDIS_HOST" in os.environ:
            host = os.environ["REDIS_HOST"]
        else:
            host = keyring.get_password(
                "upstash-confirm-coordinator-host", os.environ["USER"]
            )
    if port is None:
        if "REDIS_PORT" in os.environ:
            port = os.environ["REDIS_PORT"]
        else:
            port = 37085
    if password is None:
        if "REDIS_PASSWORD" in os.environ:
            password = os.environ["REDIS_PASSWORD"]
        else:
            password = keyring.get_password(
                "upstash-confirm-coordinator-password", os.environ["USER"]
            )
    return dict(host=host, port=port, password=password)


def get_ch_config(host=None, port=None, username=None, password=None, database=None):
    if host is None:
        if "CLICKHOUSE_HOST" in os.environ:
            host = os.environ["CLICKHOUSE_HOST"]
        else:
            host = keyring.get_password(
                "clickhouse-confirm-test-host", os.environ["USER"]
            )
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
        if "CLICKHOUSE_PASSWORD" in os.environ:
            password = os.environ["CLICKHOUSE_PASSWORD"]
        else:
            password = keyring.get_password(
                "clickhouse-confirm-test-password", os.environ["USER"]
            )
    print(f"Connecting to {host}:{port}/{database} as {username}.")
    return dict(
        host=host, port=port, username=username, password=password, database=database
    )


def find_unique_job_id(client, attempts=3):
    for i in range(attempts):
        job_id = uuid.uuid4().hex
        query = (
            "select * from information_schema.schemata"
            f"  where schema_name = '{job_id}'"
        )
        df = client.query_df(query)
        if df.shape[0] == 0:
            break

        print("OMG WOW UUID COLLISION. GO TELL YOUR FRIENDS.")
        if i == attempts - 1:
            raise Exception(
                "Could not find a unique job id." " This should never happen"
            )
    client.command(f"create database {job_id}")
    return job_id


def clear_dbs(client):
    """
    DANGER, WARNING, ACHTUNG, PELIGRO:
        Don't run this function for our production Clickhouse server. That
        would be bad. There's a built-in safeguard to prevent this, but it's
        not foolproof.

    Clear all databases (and database tables) from the Clickhouse server. This
    should only work for our test database or for localhost.

    Args:
        client: _description_
    """
    test_host = keyring.get_password("clickhouse-confirm-test-host", os.environ["USER"])
    if not (
        (test_host is not None and test_host in client.url) or "localhost" in client.url
    ):
        raise RuntimeError("This function is only for localhost or test databases.")

    to_drop = []
    for db in client.query_df("show databases")["name"]:
        if db not in ["default", "INFORMATION_SCHEMA", "information_schema", "system"]:
            to_drop.append(db)
    if len(to_drop) == 0:
        print("No databases to drop.")
        return

    print("Dropping the following databases:")
    print(to_drop)
    print("Are you sure? [yN]")
    if input() == "y":
        for db in to_drop:
            cmd = f"drop database {db}"
            print(cmd)
            client.command(cmd)
