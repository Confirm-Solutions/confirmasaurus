```python
import clickhouse_connect
import os
import keyring

host = keyring.get_password("clickhouse-confirm-host", os.environ["USER"])
password = keyring.get_password("clickhouse-confirm-password", os.environ["USER"])
client = clickhouse_connect.get_client(
    host=host,
    port=8443,
    username="default",
    password=password,
)
# client = clickhouse_connect.get_client(host='127.0.0.1', port=8123)
```

```python
client.command("CREATE TABLE test_table (key UInt16, value String) ENGINE Memory")
data = [[100, "value1"], [200, "value2"]]
client.insert("test_table", data)
print(client.query("SELECT * FROM test_table").result_set)
```

```python
import pandas as pd

df = pd.read_parquet("./dbtest.parquet")
```

```python
df[:2000000].to_parquet("./dbtestsmall.parquet")
```

```python
query = (
    "create table tiles ("
    + ",".join([f"{c} Float32" for c in df.columns])
    + ") engine = MergeTree() order by orig_lam"
)
query
```

```python
client.command(query)
```

```python
%%time
client.insert_df("tiles", df[:100000])
```

```python
import pyarrow
```

```python
%%time
client.insert_arrow("tiles", pyarrow.Table.from_pandas(df[:100000]))
```

```python
%%time
sel_df = client.query_df("select * from tiles order by orig_lam limit 10000")
sel_df.shape
```

```python
%%time
client.query(
    "select " + ",".join([f"min(B_lam_{i})" for i in range(50)]) + " from tiles"
).result_set
```

```python
%%time
client.query_df("select count(*) from tiles")
```

```python
import time

start = time.time()
i = 0
while time.time() - start < 60 * 60:
    frac = 10
    q1 = client.query_df(
        f"select count(*) from tiles where sqrt (power(theta_0+0.5,2)+power(theta_1+0.5,2)+power(theta_2+0.5,2)+power(theta_3+0.5,2)) < {frac}"
    )
    q2 = client.query(
        "select " + ",".join([f"min(B_lam_{i})" for i in range(50)]) + " from tiles"
    ).result_set
    if i % 60 == 0:
        print(i, q1, q2)
    i += 1
```

```python
# 393.75 at 6:24, wait a bit
(495 - 393.75) * 0.00239
```

```python
usage = 1680 - 828.75
price = 0.00239
time = 60
estimated_cu_per_min = usage / time
```

## Materialized View

```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host=host, port=8443, username="default", password=password
)
```

```python
client.command("drop view Blamss")
```

```python
query = (
    "create materialized view if not exists Blamss order by (Blamss0) POPULATE as select "
    + ",".join([f"min(B_lam_{i}) as Blamss{i}" for i in range(50)])
    + " from tiles"
)
```

```python
client.command(query)
```

```python
%%time
client.query(
    "select " + ",".join([f"min(Blamss{i})" for i in range(50)]) + " from Blamss"
).result_set
```

```python
%%time
client.insert_df("tiles", df[:100000])
```

## Add index

```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host=host,
    port=8443,
    username="default",
    password=password,
)
```

```python
%%time
client.query("select min(B_lam_0) from tiles")
```

```python
for i in range(50):
    client.query(f"alter table tiles drop index Blam{i}idx")
```

```python
for i in range(1):
    client.query(
        f"alter table tiles add index Blam{i}idx B_lam_{i} TYPE minmax GRANULARITY 1"
    )
for i in range(1):
    client.query(f"alter table tiles materialize index Blam{i}idx")
```

```python
%%time
client.query("select min(B_lam_0) from tiles")
```

```python
%%time
bl0 = client.query("select B_lam_0 from tiles limit 1").result_set[0][0]
```

```python
bl0
```

```python
%%time
client.query(
    f"select min(B_lam_0) from tiles where B_lam_0 < 0.063",
    settings={"force_data_skipping_indices": "Blam0idx"},
).result_set
```

```python
%%time
client.query(
    "select min(B_lam_0) from tiles",
    settings={"force_data_skipping_indices": "Blam0idx"},
).result_set
```

```python
%%time
client.query(
    "select " + ",".join([f"min(B_lam_{i})" for i in range(50)]) + " from tiles"
).result_set
```

## Test locking mechanism

janky locking mechanisms from https://blog.qryn.dev/shared-resource-lock-powered-by-clickhouse

the comments suggest `query_id`. this doesn't work for me.

```python
clickhouse_connect
```

```python
client.command("drop table _mtx")
```

```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host=host, port=8443, username="default", password=password
)
```

```python
import time
import clickhouse_connect
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool

query = "select " + ",".join([f"min(Blamss{i})" for i in range(50)]) + " from Blamss"

client.command("drop table if exists _mtx0", settings={"wait_end_of_query": 1})


@contextmanager
def claim_ch_mutex(client, mutex_id, timeout=60, claim_interval=60, retry_interval=0.5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            client.command(
                f"CREATE TABLE _mtx{mutex_id} (creation_time int, claim_interval int) order by creation_time"
            )
            client.insert(f"_mtx{mutex_id}", [[int(time.time()), claim_interval]])
            print("created and inserted")
            break
        except clickhouse_connect.driver.exceptions.DatabaseError as e:
            if f"_mtx{mutex_id} already exists" not in str(e):
                raise
            try:
                mtx_df = client.query_df(f"select * from _mtx{mutex_id}")
            except clickhouse_connect.driver.exceptions.DatabaseError as e:
                if f"_mtx{mutex_id} doesn't exist" in str(e):
                    print("table doesn't exist now! retrying")
                    continue
                raise
            if mtx_df.shape[0] > 0:
                mtx = mtx_df.iloc[0]
                if time.time() - mtx["creation_time"] > mtx["claim_interval"]:
                    print(f"Mutex{mutex_id} claim expired, deleting claim.")
                    client.command("DROP TABLE _mtx")
            print(
                f"Mutex{mutex_id} is locked. Sleeping for {retry_interval}s and then retrying."
            )
            time.sleep(retry_interval)
    if time.time() - start > timeout:
        raise Exception(f"Could not claim mutex{mutex_id} within {timeout}s")
    try:
        yield
    finally:
        client.command(f"DROP TABLE _mtx{mutex_id}")


def connect_and_query(worker_id):
    client = clickhouse_connect.get_client(
        host=host, port=8443, username="default", password=password
    )
    with claim_ch_mutex(client, 0):
        return client.query_df(query)


N = 4
ThreadPool(N).map(lambda x: connect_and_query(x), range(N))
```

```python

```
