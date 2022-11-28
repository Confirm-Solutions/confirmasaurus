```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host="jx5gomuqu1.us-east-1.aws.clickhouse.cloud",
    port=8443,
    username="default",
    password="LujWuvKhdIKn",
)
# client = clickhouse_connect.get_client(host='127.0.0.1', port=8123)
```

```python
client.command("CREATE TABLE test_table (key UInt16, value String) ENGINE Memory")
```

```python
data = [[100, "value1"], [200, "value2"]]
client.insert("test_table", data)
print(client.query("SELECT * FROM test_table").result_set)
```

```python
import pandas as pd

df = pd.read_parquet("./dbtest.parquet")
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
client.insert_df("tiles", df)
```

```python
%%time
client.insert_df("tiles", df[:400000])
```

```python
import pyarrow

tbl = pyarrow.Table.from_pandas(df[:5000000])
```

```python
%%time
client.insert_arrow("tiles", tbl)
```

```python
df.shape[0] / (7 * 60 + 43.8)
```

```python
%%time
sel_df = client.query_df("select * from tiles order by orig_lam limit 10000")
sel_df.shape
```

```python
sel_df.shape[0] * sel_df.shape[1] * 4 / 1e6
```

```python
%%time
client.query(
    "select " + ",".join([f"min(B_lam_{i})" for i in range(50)]) + " from tiles"
).result_set
```
