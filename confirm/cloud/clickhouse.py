import asyncio
import logging
import os
import time
import uuid

import pyarrow

from ..adagrid.db import DuckDBTiles

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


async def backup(ch_db, duck):
    await asyncio.gather(*[backup_table(duck, ch_db, name) for name in all_tables])


async def restore(ch_db, duck=None):
    if duck is None:
        duck = DuckDBTiles.connect()
    await asyncio.gather(*[restore_table(duck, ch_db, name) for name in all_tables])
    return duck


async def backup_table(duck, ch_client, name):
    if not duck.does_table_exist(name):
        logger.info(
            f"Backup skipping table {name} because it"
            " doesn't exist in the source db."
        )
        return
    df = duck.con.query(f"select * from {name}").df()

    def move_table_to_ch():
        cols = get_create_table_cols(df)
        if does_table_exist(ch_client, name):
            command(ch_client, f"DROP TABLE {name}")
        command(
            ch_client,
            f"""
            CREATE TABLE {name} ({",".join(cols)})
            ENGINE = MergeTree()
            ORDER BY ()
            """,
        )
        insert_df(ch_client, name, df)

    await asyncio.to_thread(move_table_to_ch)


async def restore_table(duck, ch_client, name):
    def get_table_from_ch():
        if not does_table_exist(ch_client, name):
            logger.info(
                f"Restore skipping table {name} because it"
                " doesn't exist in the source db."
            )
            return
        df = query_df(ch_client, f"select * from {name}")
        if name == "logs":
            df["t"] = df["t"].dt.tz_localize(None)
        return df

    df = await asyncio.to_thread(get_table_from_ch)  # noqa
    if duck.does_table_exist(name):
        duck.con.execute(f"drop table {name}")
    duck.con.execute(f"create table {name} as select * from df")


def does_table_exist(client, table_name: str) -> bool:
    return (
        len(
            query(
                client,
                f"""
        select * from information_schema.tables
            where table_schema = '{client.database}'
                and table_name = '{table_name}'
        """,
            ).result_set
        )
        > 0
    )


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


def query_df(client, query):
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


def insert_df(client, table, df, settings=None):
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


def query(client, query, *args, **kwargs):
    start = time.time()
    out = retry_ch_action(client.query, query, *args, **kwargs)
    logger.debug(f"Query took {time.time() - start} seconds\n{query} ")
    return out


def command(client, query, *args, **kwargs):
    start = time.time()
    out = retry_ch_action(client.command, query, *args, **kwargs)
    logger.debug(f"Command took {time.time() - start} seconds\n{query}")
    return out


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
        command(
            client,
            f"create database if not exists {job_id}",
            settings=default_insert_settings,
        )

    # NOTE: client.database is invading private API, but based on reading
    # the clickhouse_connect code, this is unlikely to break
    client.database = job_id

    logger.info(f"Connected to job {job_id}")
    return client


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
            if db.startswith("unnamed_"):
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
                command(ch_client, cmd)
