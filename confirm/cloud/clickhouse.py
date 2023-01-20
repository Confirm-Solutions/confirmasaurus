"""
Some notes I don't want to lose:

Speeding up clickhouse stuff by using threading and specifying column_types:
    https://clickhousedb.slack.com/archives/CU478UEQZ/p1673560678632889

Fast min/max with clickhouse:
    https://clickhousedb.slack.com/archives/CU478UEQZ/p1669820710989079
"""
import os
import time
import uuid
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict
from typing import List

import clickhouse_connect
import numpy as np

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
from confirm.adagrid.convergence import WorkerStatus

import imprint.log

logger = imprint.log.getLogger(__name__)

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


def distributed_next(
    db, convergence_f, n_steps, step_size, packet_size, order_col, worker_id
):
    """
    This is a distributed version of the next function. It is used to
    distribute the tile processing work.

    This is the crux of the distributed version of the algorithm.
    """
    # If we loop 25 times, there's definitely a bug.
    max_loops = 25
    i = 0
    status = WorkerStatus.WORKING
    report = dict()
    while i < max_loops:
        report["distributed_next_loop"] = i
        with db.lock:
            step_id, step_iter, step_n_iter = db._get_step_info()
            report["step_id"] = step_id
            report["step_iter"] = step_iter
            report["step_n_iter"] = step_n_iter

            # step_id = None means that we're starting a new job and haven't
            # yet created a step.
            if step_iter < step_n_iter:
                logger.debug(f"get_work(step_id={step_id}, step_iter={step_iter})")
                work = db.get_work(
                    step_id, step_iter, packet_size, order_col, worker_id
                )
                report["work_extraction_time"] = time.time()
                logger.debug(f"get_work(...) returned {work.shape[0]} tiles")

                # If there's work, return it!
                if work.shape[0] > 0:
                    db._set_step_info((step_id, step_iter + 1, step_n_iter))
                    return status, work, report

            # If there's no work, we check if the step is complete.
            if n_tiles_left_in_step(db, step_id) == 0:
                # If a packet has just been completed, we check for convergence.
                status = convergence_f()
                if status:
                    return WorkerStatus.CONVERGED, None, report

                # If we haven't converged, we create a new packet.
                if step_id >= n_steps - 1:
                    # We've finished all the steps, so we're done.
                    return WorkerStatus.REACHED_N_STEPS, None, report

                new_step_id = new_step(db, step_id, step_size, packet_size, order_col)

                if new_step_id == "empty":
                    # New packet is empty so we have finished but failed to converge.
                    return WorkerStatus.FAILED, None, report
                else:
                    # Successful new packet. We should check for work again
                    # immediately.
                    status = WorkerStatus.NEW_STEP
                    wait = False
            else:
                # No work available, but the packet is incomplete. We should
                # release the lock and wait for other workers to finish.
                wait = True
        if wait:
            # TODO: the sleep time should be configurable.
            time.sleep(1)
        if i > 2:
            logger.warning(
                f"Worker {worker_id} has been waiting for work for"
                f" {i} iterations. This might indicate a bug."
            )
        i += 1
    return WorkerStatus.STUCK, None, report


def n_tiles_left_in_step(db, step_id):
    return db.client.query(
        f"""
        select count(*) from tiles_step
            where step_id = {step_id}
                and id not in (
                    select id from tiles_done where step_id = {step_id}
                )
        """
    ).result_set[0][0]


def new_step(db, old_step_id, step_size, packet_size, order_col):
    new_step_id = old_step_id + 1
    df = _query_df(
        db.client,
        f"""
        select id from tiles where
            id not in (select id from tiles_done)
        order by {order_col} asc limit {step_size}
    """,
    )
    if df.shape[0] == 0:
        return "empty"

    df["step_id"] = new_step_id

    n_tiles = df.shape[0]
    n_packets = int(np.ceil(n_tiles / packet_size))
    splits = np.array_split(np.arange(n_tiles), n_packets)
    assignment = np.empty(n_tiles, dtype=np.int32)
    for i in range(n_packets):
        assignment[splits[i]] = i
    rng = np.random.default_rng()
    rng.shuffle(assignment)

    df["step_iter"] = assignment
    logger.debug(
        f"new step {(new_step_id, 0, n_packets)} "
        f"n_tiles={n_tiles} packet_size={packet_size}"
    )
    logger.debug(f"new step df.head(): {df.head()}")
    _insert_df(db.client, "tiles_step", df)

    db._set_step_info((new_step_id, 0, n_packets))
    return new_step_id


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
        - the "tiles_done" table, which contains a column indicating which
          tiles are inactive.
        - a tile is only active is active=True and id not in tiles_inactive

    eligible: True if the tile is eligible to be selected for work. Specified in
         two ways
        - the "eligibile" column in the tiles table
        - the "tiles_work" table, which contains tiles that are ineligible
          because they either are being worked on or have been worked on
        - a tile is only eligible if eligible=True and id not in tiles_work

    The reason for these dual sources of information is the desire to only
    rarely update rows that have already been written. Updating data is painful
    in Clickhouse. We could periodically flush the tiles_ineligible and
    tiles_inactive tables to corresponding columns in the tiles table. This
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
    _columns: List[str] = None
    _d: int = None
    next = distributed_next

    def __post_init__(self):
        self.lock = redis.lock.Lock(self.redis_con, f"{self.job_id}:next_lock")

    def _get_step_info(self):
        out = literal_eval(self.redis_con.get(f"{self.job_id}:step_info").decode())
        logger.debug("get step info: %s", out)
        return out

    def _set_step_info(self, step_info):
        logger.debug("set step info: %s", step_info)
        self.redis_con.set(f"{self.job_id}:step_info", str(step_info))

    @property
    def host(self):
        return self.connection_details["host"]

    @property
    def store(self):
        return ClickhouseStore(self.client, self.job_id)

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
                and(active=true,
                    (id not in (select id from tiles_inactive)))
                    as active,
                and(eligible=true,
                    (id not in (select id from tiles_done)))
                    as eligible
            from tiles
        """,
        )

    def write(self, df):
        logger.debug(f"writing {df.shape[0]} new tiles")
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

    def get_work(self, step_id, step_iter, n, order_col, worker_id):
        out = _query_df(
            self.client,
            f"""
            select t.* from tiles_step tp
                left join tiles t on t.id = tp.id
            where
                step_id = {step_id}
                and step_iter = {step_iter}
            order by t.{order_col} asc limit {n}
        """,
        )
        out.rename(columns={"t.id": "id"}, inplace=True)
        T = time.time()
        out["time"] = T
        out["worker_id"] = worker_id
        out["step_id"] = step_id
        out["step_iter"] = step_iter
        return out

    def finish(self, which):
        write_subset = which[
            ["step_id", "step_iter", "id", "active", "time", "worker_id"]
        ]
        logger.debug(f"finish: {write_subset.head()}")
        _insert_df(self.client, "tiles_done", write_subset)

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
        # tables:
        # - tiles: id, lots of other stuff...
        # - packet: id
        # - work: id
        # - done: id, active
        # - inactive: materialized view based on done
        cols = get_create_table_cols(df)
        orderby = "orderer" if "orderer" in df.columns else "id"
        id_type = type_map[df["id"].dtype.name]
        commands = [
            f"""
            create table tiles ({",".join(cols)})
                engine = MergeTree() order by {orderby}
            """,
            f"""
            create table tiles_step (step_id UInt32, step_iter UInt32, id {id_type})
                engine = MergeTree() order by (step_id, step_iter, id)
            """,
            f"""
            create table tiles_done (
                    step_id UInt32,
                    step_iter UInt32,
                    id {id_type},
                    active Bool,
                    time Float64,
                    worker_id UInt32)
                engine = MergeTree() order by (step_id, step_iter, id)
            """,
            """
            create materialized view tiles_inactive
                engine = MergeTree() order by id
                as select id from tiles_done where active=false
            """,
        ]
        for c in commands:
            self.client.command(c)
        self.write(df)
        self._set_step_info((-1, 0, 0))

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
            host: keyring entry "clickhouse-confirm-test-host"
            port: 8443
            username: "default"
            password: keyring entry "clickhouse-confirm-test-password"

        For Redis, we will use the following environment variables:
            REDIS_HOST: The hostname for the Redis server.
            REDIS_PORT: The Redis server port.
            REDIS_PASSWORD: The Redis password.

        If the environment variables are not set, the defaults will be:
            host: keyring entry "upstash-confirm-coord-test-host"
            port: 37085
            password: keyring entry "upstash-confirm-coord-test-password"

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
            test_host = get_ch_test_host()
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


def get_redis_client(host=None, port=None, password=None):
    return redis.Redis(**get_redis_config(host, port, password))


def get_redis_config(host=None, port=None, password=None):
    if host is None:
        if "REDIS_HOST" in os.environ:
            host = os.environ["REDIS_HOST"]
        else:
            host = keyring.get_password(
                "upstash-confirm-coord-test-host", os.environ["USER"]
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
                "upstash-confirm-coord-test-password", os.environ["USER"]
            )
    return dict(host=host, port=port, password=password)


def get_ch_client(host=None, port=None, username=None, password=None, job_id=None):
    connection_details = get_ch_config(host, port, username, password, job_id)
    return clickhouse_connect.get_client(**connection_details)


def get_ch_config(host=None, port=None, username=None, password=None, database=None):
    if host is None:
        if "CLICKHOUSE_HOST" in os.environ:
            host = os.environ["CLICKHOUSE_HOST"]
        else:
            host = get_ch_test_host()
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
    logger.info(f"Clickhouse config: {username}@{host}:{port}/{database}")
    return dict(
        host=host, port=port, username=username, password=password, database=database
    )


def get_ch_test_host():
    if "CLICKHOUSE_TEST_HOST" in os.environ:
        return os.environ["CLICKHOUSE_TEST_HOST"]
    else:
        return keyring.get_password("clickhouse-confirm-test-host", os.environ["USER"])


def clear_dbs(ch_client, redis_client, names=None, yes=False):
    """
    DANGER, WARNING, ACHTUNG, PELIGRO:
        Don't run this function for our production Clickhouse server. That
        would be bad. There's a built-in safeguard to prevent this, but it's
        not foolproof.

    Clear all databases (and database tables) from the Clickhouse server. This
    should only work for our test database or for localhost.

    Args:
        client: Clickhouse client
        names: default None, list of database names to drop. If None, drop all.
        yes: bool, if True, don't ask for confirmation
    """
    test_host = get_ch_test_host()
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
        print("No databases to drop.")
        return

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
            keys = list(redis_client.scan_iter("*{db}*"))
            print("deleting redis keys", keys)
            redis_client.delete(*keys)
