```python
import pandas as pd

df = pd.read_parquet("../clickhouse/dbtest.parquet")
```

```python
import snowflake.connector

password = ""
account_name = ""
username = "tbenthompson"
```

```python
# con = snowflake.connector.connect(
#     user=username,
#     password=password,
#     account=account_name
# )
# con.cursor().execute("""
#     CREATE WAREHOUSE IF NOT EXISTS imprint_warehouse WITH
#         WAREHOUSE_SIZE = 'X-SMALL'
#         WAREHOUSE_TYPE = 'STANDARD'
#         AUTO_SUSPEND = 60
#         AUTO_RESUME = TRUE
#         MIN_CLUSTER_COUNT = 1
#         MAX_CLUSTER_COUNT = 1
#         INITIALLY_SUSPENDED = TRUE
#         COMMENT = 'Imprint Warehouse';
# """)
# con.cursor().execute("""
#     CREATE DATABASE IF NOT EXISTS imprint_db;
# """)
df_chunk = df[:1000].rename(lambda x: x.upper(), axis=1)
df_chunk.shape
```

```python
import snowflake.connector.pandas_tools
```

```python
from sqlalchemy import create_engine

engine = create_engine(
    f"snowflake://{username}:{password}@{account_name}/imprint_db/public?warehouse=imprint_warehouse"
)
```

```python
df_chunk.columns
```

```python
%%timeit
df_chunk.to_sql(
    "TILES",
    engine,
    index=False,
    method=snowflake.connector.pandas_tools.pd_writer,
    if_exists="append",
)
```

```python
con = snowflake.connector.connect(
    user=username,
    password=password,
    account=account_name,
    database="imprint_db",
    warehouse="imprint_warehouse",
    schema="public",
)
```

```python
%%timeit
con.cursor().execute("select count(*) from tiles").fetchall()
```

```python
%%time
con.cursor().execute("select 1;").fetchall()
```

```python
con.commit()
```

```python
df_chunk = df[:1000000].rename(lambda x: x.upper(), axis=1)
```

```python
%%time
success, nchunks, nrows, output = snowflake.connector.pandas_tools.write_pandas(
    con, df_chunk, "TILES", database="IMPRINT_DB", schema="PUBLIC", compression="gzip"
)
success, nchunks, nrows
```

```python

```
