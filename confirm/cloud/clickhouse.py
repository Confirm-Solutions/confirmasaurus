import os
import time
import uuid
from dataclasses import dataclass
from typing import List

import clickhouse_connect
import keyring
import pottery
import pyarrow
import redis

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
    return [f"{c} {type_map[dt.name]}" for c, dt in zip(df.columns, df.dtypes)]


def _table_list(client):
    result = client.query_df("show tables")
    if result.shape[0] == 0:
        return []
    else:
        return result["name"].values


@dataclass
class Clickhouse:
    """
    A tile database built on top of Clickhouse. This should be very fast and
    robust and is preferred for large runs.

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
    in Clickhouse. It's currently unimplemented, but we could periodically
    flush the tiles_ineligible and tiles_inactive tables to corresponding
    columns in the tiles table. This would reduce the size of the subqueries
    and probably improve performance for very large jobs.

    NOTE: A tile may be active but ineligible if it was selected for work but the
    criterion decided it was not worth refining or deepening.
    """

    ch_client: clickhouse_connect.driver.httpclient.HttpClient
    redis_con: redis.Redis
    host: str
    job_id: str
    worker_id: int
    _columns: List[str] = None
    _d: int = None

    def _query_df(self, query):
        return self.ch_client.query_arrow(query).to_pandas()

    def _insert_df(self, table, df):
        self.ch_client.insert_arrow(
            table, pyarrow.Table.from_pandas(df, preserve_index=False)
        )

    ########################################
    # Caching interface
    ########################################
    def load(self, table_name):
        return self._query_df(f"select * from {table_name}")

    def store(self, table_name, df):
        if table_name not in _table_list(self.ch_client):
            if "id" not in df.columns:
                df["id"] = df.index
            cols = get_create_table_cols(df)
            self.ch_client.command(
                f"""
                create table {table_name} ({",".join(cols)})
                    engine = MergeTree() order by id
            """
            )
        self._insert_df(table_name, df)

    ########################################
    # Adagrid tile database interface
    ########################################
    def dimension(self):
        if self._d is None:
            self._d = (
                max([int(c[5:]) for c in self.columns() if c.startswith("theta")]) + 1
            )
        return self._d

    def columns(self):
        if self._columns is None:
            self._columns = self._query_df("select * from tiles limit 1").columns
        return self._columns

    def get_all(self):
        cols = ",".join([c for c in self.columns() if c not in ["active", "eligible"]])
        return self._query_df(
            f"""
            select {cols},
                and(active=true, (id not in (select id from tiles_inactive))) as active,
                and(eligible=true, (id not in (select id from work))) as eligible
            from tiles
        """
        )

    def write(self, df):
        self._insert_df("tiles", df)

    def worst_tile(self, order_col):
        return self._query_df(
            f"""
            select * from tiles where
                active=true
                and id not in (select * from tiles_inactive)
            order by {order_col} limit 1
        """
        )

    def next(self, n, order_col):
        lock = pottery.Redlock(
            key=f"{self.host}/{self.job_id}/lock", masters={self.redis_con}
        )
        with lock:
            out = self._query_df(
                f"""
                select * from tiles where
                    eligible=true
                    and id not in (select id from work)
                order by {order_col} asc limit {n}
            """
            )
            T = time.time()
            out["time"] = T
            out["worker_id"] = self.worker_id
            self._insert_df("work", out[["id", "time", "worker_id"]])

        # No need to check for conflicts because we are using a lock but if we
        # were to check for conflicts, it would look something like this
        # conflicts = self._query_df(
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
        self._insert_df("tiles_inactive", which.loc[~which["active"]][["id"]])

    def bootstrap_lamss(self):
        # Get the number of bootstrap lambda* columns
        nB = max([int(c[6:]) for c in self.columns() if c.startswith("B_lams")]) + 1

        # Get lambda**_Bi for each bootstrap sample.
        cols = ",".join([f"min(B_lams{i})" for i in range(nB)])
        lamss = self.ch_client.query(
            f"""
            select {cols} from tiles where
                active=true
                and id not in (select id from tiles_inactive)
        """
        ).result_set[0]

        return lamss

    def close(self):
        self.ch_client.close()

    def init_tiles(self, df):
        cols = get_create_table_cols(df)
        orderby = "orderer" if "orderer" in df.columns else "id"
        self.ch_client.command(
            f"""
            create table tiles ({",".join(cols)})
                engine = MergeTree() order by {orderby}
        """
        )

        id_type = type_map[df["id"].dtype.name]
        self.ch_client.command(
            f"""
            create table tiles_inactive (id {id_type})
                engine = MergeTree() order by id
            """
        )
        self.ch_client.command(
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
        if job_id is None:
            client = clickhouse_connect.get_client(
                **get_ch_config(host, port, username, password)
            )
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
