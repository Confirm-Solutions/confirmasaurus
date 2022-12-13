import os
import time
import uuid
from dataclasses import dataclass
from typing import List

import clickhouse_connect
import keyring
import pyarrow


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

    client: clickhouse_connect.driver.httpclient.HttpClient
    job_id: str
    worker_id: int
    _columns: List[str] = None

    def _query_df(self, query):
        return self.client.query_arrow(query).to_pandas()

    def _insert_df(self, table, df):
        self.client.insert_arrow(
            table, pyarrow.Table.from_pandas(df, preserve_index=False)
        )

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

        conflicts = self._query_df(
            f"""
            with mine as (
                select id, time from work where
                    worker_id={self.worker_id}
                    and time={T}
            )
            select worker_id, time from work
                join mine on (
                    work.id=mine.id
                    and work.worker_id != {self.worker_id}
                )
                where work.time <= mine.time
        """
        )
        assert conflicts.shape[0] == 0
        return out

    def finish(self, which):
        self._insert_df("tiles_inactive", which.loc[~which["active"]][["id"]])

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
        types = {
            "uint32": "UInt32",
            "uint64": "UInt64",
            "float32": "Float32",
            "float64": "Float64",
            "int32": "Int32",
            "int64": "Int64",
            "bool": "Boolean",
            "string": "String",
            "bytes640": "FixedString(80)",
        }
        cols = [f"{c} {types[dt.name]}" for c, dt in zip(df.columns, df.dtypes)]
        id_type = types[df["id"].dtype.name]

        orderby = "orderer" if "orderer" in df.columns else "id"

        self.client.command(
            f"""
            create table tiles ({",".join(cols)})
                engine = MergeTree() order by {orderby}
        """
        )
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
        worker_id: int = 0,
        host=None,
        port=None,
        username=None,
        password=None,
    ):
        if job_id is None:
            client = get_client(host, port, username, password)
            job_id = find_unique_job_id(client)
        client = get_client(host, port, username, password, job_id)
        return Clickhouse(client, job_id, worker_id)


def get_client(host=None, port=None, username=None, password=None, database=None):
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
    return clickhouse_connect.get_client(
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
            client.command(f"drop database {db}")
