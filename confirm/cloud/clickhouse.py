import asyncio
import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass
from dataclasses import field
from pprint import pprint
from typing import Any
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow

if TYPE_CHECKING:
    import clickhouse_connect

from ..adagrid.db import SQLTiles

logger = logging.getLogger(__name__)


async_insert_settings = {"async_insert": 1, "wait_for_async_insert": 0}
sync_insert_settings = dict(
    insert_distributed_sync=1, insert_quorum=2, insert_quorum_parallel=1
)
select_settings = dict(
    select_sequential_consistency=1, load_balancing="first_or_random"
)


all_tables = ["results", "tiles", "done", "config", "logs", "reports"]


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
    out = []
    for c, dt in zip(df.columns, df.dtypes):
        ch_type = type_map[dt.name]
        if not (("Float" in ch_type) or ("Int" in ch_type) or ("Boolean" in ch_type)):
            ch_type = f"Nullable({ch_type})"
        out.append(f"{c} {ch_type}")
    return out


def create_table(ch_client, name, df, partitioner, orderer):
    cols = get_create_table_cols(df)
    command(
        ch_client,
        f"""
        CREATE TABLE IF NOT EXISTS {name} ({",".join(cols)})
        ENGINE = MergeTree()
        PARTITION BY ({partitioner})
        ORDER BY ({orderer})
        """,
        sync_insert_settings,
    )


def should_block(block_requested):
    # Sometimes this gets called after the event loop has been closed. In that
    # case, we run the insert directly rather than using asyncio.
    if not block_requested:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                'Running blocking insert despite "block" '
                "being False because there is no running event loop."
            )
            block_requested = True
    return block_requested


def insert_df(ch_client, name, df, settings=None, chunk_size=30000, block: bool = True):
    if should_block(block):

        def wrapper(*args):
            args[0](*args[1:])

    else:

        def wrapper(*args):
            return asyncio.create_task(asyncio.to_thread(*args))

    wait_for = []
    for i in range(0, len(df), chunk_size):
        wait_for.append(
            wrapper(
                _insert_chunk_df, ch_client, name, df.iloc[i : i + chunk_size], settings
            )
        )
    return wait_for


def _insert_chunk_df(client, table, df, settings=None):
    # Same as _query_df, inserting through arrow is faster!
    start = time.time()
    retry_ch_action(
        client.insert_arrow,
        table,
        pyarrow.Table.from_pandas(df, preserve_index=False),
        settings=settings or sync_insert_settings,
    )
    logger.debug(
        f"Inserting {df.shape[0]} rows into {table} took"
        f" {time.time() - start} seconds"
    )


async def restore(ch_client, duck):
    await asyncio.gather(*[restore_table(ch_client, duck, name) for name in all_tables])
    duck.update_active_complete()


async def restore_table(ch_client, duck, name):
    # TODO: might need offset/limit and iteration here.
    def get_table_from_ch():
        cols = "*"
        if name == "results":
            col_list = query_df(ch_client, f"select * from {name} limit 0").columns
            cols = ",".join([c for c in col_list if "twb_lams" not in c])
        df = query_df(ch_client, f"select {cols} from {name}")
        if name == "logs":
            df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None)
        return df

    df = await asyncio.to_thread(get_table_from_ch)  # noqa
    if duck.does_table_exist(name):
        duck.con.execute(f"drop table {name}")
    duck.con.execute(f"create table {name} as select * from df")


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


def query_df(client, query, quiet=False, **kwargs):
    # Loading via Arrow and then converting to Pandas is faster than using
    # query_df directly to load a pandas dataframe. I'm guessing that loading
    # through arrow is a little less flexible or something, but for our
    # purposes, faster loading is great.
    start = time.time()
    out = retry_ch_action(client.query_arrow, query, **kwargs).to_pandas()
    if not quiet:
        logger.debug(f"Query took {time.time() - start} seconds\n{query}")
    return out


def query(client, query, quiet=False, *args, **kwargs):
    start = time.time()
    out = retry_ch_action(client.query, query, *args, **kwargs)
    if not quiet:
        logger.debug(f"Query took {time.time() - start} seconds\n{query} ")
    return out


def command(client, query, *args, **kwargs):
    start = time.time()
    out = retry_ch_action(client.command, query, *args, **kwargs)
    logger.debug(f"Command took {time.time() - start} seconds\n{query}")
    return out


def get_ch_client(
    connection_details=None,
    service="TEST",
    host=None,
    port=None,
    username=None,
    password=None,
    database=None,
    **kwargs,
):
    import clickhouse_connect

    clickhouse_connect.common.set_setting("autogenerate_session_id", False)
    if connection_details is None:
        connection_details = get_ch_config(
            service, host, port, username, password, database
        )
    connection_details["connect_timeout"] = 30
    # NOTE: this is a way to run more than the default 8 queries at once.
    if "pool_mgr" not in kwargs:
        kwargs["pool_mgr"] = clickhouse_connect.driver.httputil.get_pool_manager(
            maxsize=16, block=True
        )
    connection_details.update(kwargs)
    return clickhouse_connect.get_client(**connection_details)


def get_ch_config(
    service="TEST", host=None, port=None, username=None, password=None, database=None
):
    if host is None:
        host = os.environ[f"CLICKHOUSE_{service}_HOST"]
    if port is None:
        if f"CLICKHOUSE_{service}_PORT" in os.environ:
            port = os.environ[f"CLICKHOUSE_{service}_PORT"]
        else:
            port = 8443
    if username is None:
        if f"CLICKHOUSE_{service}_USERNAME" in os.environ:
            username = os.environ[f"CLICKHOUSE_{service}_USERNAME"]
        else:
            username = "default"
    if password is None:
        password = os.environ[f"CLICKHOUSE_{service}_PASSWORD"]
    logger.info(f"Clickhouse config: {username}@{host}:{port}/{database}")
    return dict(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
    )


def list_dbs(ch_client):
    return [
        row["name"]
        for i, row in retry_ch_action(ch_client.query_df, "show databases").iterrows()
        if row["name"]
        not in ["system", "default", "information_schema", "INFORMATION_SCHEMA"]
    ]


def clear_dbs(
    ch_client=None, list=False, prefix="unnamed", names=None, yes=False, exclude=None
):
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

    all_dbs = list_dbs(ch_client)
    if list:
        print("All databases: ")
        pprint(all_dbs)

    if names is None:
        print('Dropping databases starting with "{}"'.format(prefix))
        to_drop = [db for db in all_dbs if db.startswith(prefix)]
    else:
        print("dropping specified databases: {}".format(names))
        to_drop = names

    explicitly_excluded = []
    for e in exclude:
        if e in to_drop:
            to_drop.remove(e)
            explicitly_excluded.append(e)

    to_drop.sort()
    explicitly_excluded.sort()
    if len(to_drop) == 0:
        print("No Clickhouse databases to drop.")
    else:
        print("Dropping the following databases:")
        print(to_drop)
        print("Excluding the following databases:")
        print(explicitly_excluded)
        if not yes:
            print("Are you sure? [yN]\nResponse: ", flush=True, end="")
            yes = input() == "y"

        if yes:
            for db in to_drop:
                cmd = f"drop database {db}"
                print(cmd)
                command(ch_client, cmd)


# Lots of remaining performance improvements:
# - [ ] need to be able to throw out large portions of the results table. there
# are two pieces to this.
# 	- [ ] need to permanently apply the done filter to the results table and
# 	remove rows that are irrelevant.
# 	- [ ] algorithm for guessing at a conservative orderer_max and filter the
# 	query by that value!
# - [ ] the B_lams minimum query can also be made much faster by filtering out tiles
# that have no chance of winning. use twb_min_lams and twb_max_lams.
# - [ ] the lams query can use a separate table with just id and lams and use
# the index on that?? same with the twb_mean_lams query.
#
# https://engineering.contentsquare.com/2022/tools-to-analyse-slow-query-clickhouse/
# https://www.slideshare.net/Altinity/clickhouse-query-performance-tips-and-tricks-by-robert-hodges-altinity-ceo


@dataclass
class ClickhouseTiles(SQLTiles):
    connection_details: Dict[str, str]
    client: "clickhouse_connect.driver.httpclient.HttpClient"
    job_name: str
    tasks: List[asyncio.Task] = field(default_factory=list)
    expected_counts: Dict[int, Dict[str, int]] = field(default_factory=dict)
    max_done_step: int = -1
    _threadpool: concurrent.futures.ThreadPoolExecutor = field(
        default_factory=concurrent.futures.ThreadPoolExecutor
    )
    _table_exists: Dict[str, bool] = field(default_factory=set)
    _table_columns: Dict[str, List[str]] = field(default_factory=dict)
    _done_task: asyncio.Task = None
    _update_task: asyncio.Task = None
    _verify_task: asyncio.Task = None

    def list_tables(client):
        return query(
            client,
            f"""
            select * from information_schema.tables
                where table_schema = '{client.database}'
            """,
        ).result_set

    def launch_thread(self, f, *args, **kwargs):
        return asyncio.create_task(asyncio.to_thread(f, *args, **kwargs))

    def query(self, query: str, quiet: bool = False, settings=None) -> pd.DataFrame:
        if settings is None:
            settings = select_settings
        return query_df(self.client, query, quiet=quiet, settings=settings)

    def insert(
        self, table: str, df: pd.DataFrame, create: bool = True, block: bool = False
    ) -> None:
        partitioner = {
            "done": "active_at_birth, active",
            "results": "step_id",
        }

        orderer = {"done": "step_id", "results": "group_id, id"}

        def _inner():
            if create and table not in self._table_exists:
                create_table(
                    self.client,
                    table,
                    df,
                    partitioner.get(table, ""),
                    orderer.get(table, ""),
                )
                self._table_exists.add(table)

            df_subset = df[self.get_columns(table)]
            insert_df(self.client, table, df_subset, block=True)

        self.tasks.append(self._threadpool.submit(_inner))

    def get_columns(self, table_name: str) -> List[str]:
        if table_name not in self._table_columns:
            self._table_columns[table_name] = self.query(
                f"select * from {table_name} limit 0"
            ).columns
        return self._table_columns[table_name]

    def does_table_exist(self, table_name: str) -> bool:
        if table_name not in self._table_exists:
            exists = (
                self.query(
                    f"""
                        select count(*) from information_schema.tables
                            where table_schema = '{self.client.database}'
                                and table_name = '{table_name}'
                        """,
                    settings=select_settings,
                ).iloc[0][0]
                > 0
            )
            if exists:
                self._table_exists.add(table_name)
                return True
            else:
                return False
        else:
            return True

    def get_results(self) -> pd.DataFrame:
        return self.query(
            """
            select *, 
                    (id not in (
                        select id from done where active_at_birth=true and active=false
                    )) as active 
                from results
        """
        )

    def wait_for_results(self, step_id):
        self.wait_for_table(
            "Adagrid",
            "results",
            step_id,
            self.expected_counts[step_id]["results"],
            0.1,
            step_id == 0,
        )

    def wait_for_table(
        self,
        purpose: str,
        table_name: str,
        step_id: int,
        expected: int,
        sleep_time: float,
        wait_for_create: bool,
    ):
        while True:
            if wait_for_create:
                if not self.does_table_exist(table_name):
                    time.sleep(sleep_time)
                    continue

            count = self.query(
                f"""
                select count(*) from {table_name}
                where step_id = {step_id} 
                """,
                quiet=True,
            ).iloc[0][0]
            if count == expected:
                logger.debug(
                    f"{purpose} done waiting for step {step_id}: "
                    f" {count}/{expected} in {table_name}"
                )
                return
            elif count > expected:
                raise ValueError(
                    f"Too many rows in {table_name} for step {step_id}."
                    f" Expected {expected}, got {count}."
                )
            else:
                logger.debug(
                    f"{purpose} is waiting for step {step_id}: "
                    f" {count}/{expected} in {table_name}"
                )
                time.sleep(sleep_time)

    def insert_report(self, report: Dict[str, Any]) -> None:
        command(self.client, f"insert into reports values ('{json.dumps(report)}')")

    def insert_done_update_results(self, df: pd.DataFrame) -> None:
        assert self._done_task is None
        step_id = df["step_id"].iloc[0]
        insert_tasks = insert_df(
            self.client,
            "done",
            df[self.get_columns("done")],
            block=False,
        )
        insert_tasks = [t for t in insert_tasks if t is not None]

        async def wait_for_done() -> None:
            await asyncio.gather(*insert_tasks)
            return step_id

        self._done_task = asyncio.create_task(wait_for_done())

    async def prepare_step(
        self, basal_step_id: int, step_id: int, n: int, orderer: str
    ) -> None:
        self.cleanup()
        if self._done_task is not None:
            self.ready_to_update = await self._done_task
            self._done_task = None

        def min_query():
            self.wait_for_results(basal_step_id)

            cols = []
            for order_col in [
                ("lams", "Min"),
                ("twb_mean_lams", "Min"),
                ("impossible", "Max"),
            ]:
                cols.append(
                    f"arg{order_col[1]}(id, {order_col[0]}) as id{order_col[0]}"
                )
            nB = (
                max(
                    [
                        int(c[6:])
                        for c in self.get_columns("results")
                        if c.startswith("B_lams")
                    ]
                )
                + 1
            )
            cols.extend([f"min(B_lams{i})" for i in range(nB)])
            cols_str = ",".join(cols)
            return self.query(
                f"""
                select {cols_str} from results
                where step_id <= {basal_step_id}
                    and inactivation_step > {basal_step_id}
                    and (id not in (
                        select id from done
                            where
                                active_at_birth=true
                                and active = false 
                                and step_id <= {basal_step_id}
                    ))
                """
            )

        def next_query():
            self.wait_for_results(basal_step_id)
            important_cols = ",".join(
                [
                    c
                    for c in self.get_columns("results")
                    if not (c.startswith("B_lams") or c.startswith("twb_lams"))
                ]
            )
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            # TODO:
            return self.query(
                f"""
                with results_ids as (select id from results_orderer
                    where (id not in (
                            select id from done 
                                where active_at_birth=true 
                                and step_id < {step_id}
                        ))
                        and step_id <= {basal_step_id}
                order by {orderer} limit {n})
                select {important_cols} from results where id in results_ids
                """
            )

        self.min_query, self.next_query = await asyncio.gather(
            asyncio.to_thread(min_query), asyncio.to_thread(next_query)
        )

        self.ready_to_verify = basal_step_id
        self.verify_stop = False

    def next(
        self,
        basal_step_id: int,
        new_step_id: int,
        n: int,
        orderer: str,
    ) -> pd.DataFrame:
        return self.next_query

    def bootstrap_lamss(self, basal_step_id: int, nB: int) -> List[float]:
        return self.min_query.iloc[0].values[3:]

    def worst_tile(self, basal_step_id: int, order_col: str, min: bool) -> pd.DataFrame:
        tile_id = self.min_query["id" + order_col].iloc[0]
        return self.query("select * from results where id = " + str(tile_id))

    def cleanup(self) -> None:
        # Clean up finished tasks.
        wait_for_done = [t for t in self.tasks if t is not None and t.done()]
        incomplete = []
        for t in self.tasks:
            if t is None:
                continue
            if t.done():
                wait_for_done.append(t)
            else:
                incomplete.append(t)
        self.tasks = incomplete
        # By calling .result on the done Tasks, we will cause exceptions to be
        # re-raised.
        [w.result() for w in wait_for_done]
        logger.debug(
            f"Cleaned up {len(wait_for_done)} finished Clickhouse insertion tasks."
        )
        logger.debug(f"{len(self.tasks)} Clickhouse insert tasks remaining.")

    def verify(self, step_id: int, sleep_time: float = 0.5) -> asyncio.Task:
        # TRICKY!
        # In order to ensure that the tables are fully populated we need to
        # wait for inserts. But, we also need to make sure that the waiting is
        # doing *in the same thread* as the following queries in super().verify
        # This is because different threads may connect to different replicas
        # in the Clickhouse Cloud cluster.
        self.wait_for_table(
            "Verify",
            "results",
            step_id,
            self.expected_counts[step_id]["results"],
            sleep_time,
            step_id == 0,
        )
        self.wait_for_table(
            "Verify",
            "tiles",
            step_id,
            self.expected_counts[step_id]["tiles"],
            sleep_time,
            step_id == 0,
        )
        self.wait_for_table(
            "Verify",
            "done",
            step_id,
            self.expected_counts[step_id]["done"],
            sleep_time,
            step_id == 0,
        )
        super().verify(step_id)

    async def finalize(self):
        self.verify_stop = True
        self.update_stop = True
        while len(self.tasks) > 0:
            tmp = [asyncio.wrap_future(t) for t in self.tasks if t is not None]
            self.tasks = []
            await asyncio.gather(*tmp)
            if self._done_task is not None:
                await self._done_task
            if self._update_task is not None:
                await self._update_task
            if self._verify_task is not None:
                await self._verify_task

    def close(self):
        self.client.close()

    @staticmethod
    def connect(
        job_name: str,
        service="TEST",
        host=None,
        port=None,
        username=None,
        password=None,
        no_create=False,
        **kwargs,
    ):
        """
        Connect to a Clickhouse server

        Each job_name corresponds to a Clickhouse database on the Clickhouse
        cluster.

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
            job_name: The job_name.
            host: The hostname for the Clickhouse server. Defaults to None.
            port: The Clickhouse server port. Defaults to None.
            username: The Clickhouse username. Defaults to None.
            password: The Clickhouse password. Defaults to None.
            no_create: If True, do not create the job_id database. Defaults to False.

        Returns:
            A Clickhouse tile database object.
        """

        connection_details = get_ch_config(service, host, port, username, password)

        client = get_ch_client(connection_details=connection_details, **kwargs)
        if not no_create:
            # Create job_id database if it doesn't exist
            command(client, f"create database if not exists {job_name}")

        # NOTE: client.database is invading private API, but based on reading
        # the clickhouse_connect code, this is unlikely to break
        client.database = job_name

        logger.info(f"Connected to job {job_name}")
        return ClickhouseTiles(
            connection_details=connection_details, client=client, job_name=job_name
        )
