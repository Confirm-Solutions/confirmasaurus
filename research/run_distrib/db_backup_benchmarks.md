```python
import duckdb
con = duckdb.connect('../../wd41_4d_v55.db')
```

```python
con.query('select count(*) from results')
```

```python
con.execute("copy (select * from results limit 10000) to 'results.parquet' (format parquet)")
```

```python
con.execute("""
install httpfs;
load httpfs;
set s3_region='us-east-1';
set s3_access_key_id='';
set s3_secret_access_key='';
""")
```

```python
%%time
con.execute("copy (select * from results where step_id > 7 and step_id <= 10) to 'tmp.parquet' (FORMAT parquet)")
```

```python
%%time
df = con.query('select * from results where step_id > 7 and step_id <= 10').df()
df.shape[0]
```

```python
%%time
arrow_tbl = con.query('select * from results where step_id > 7 and step_id <= 10').arrow()
```

```python
%%time
df.to_csv('tmp.csv', index=False)
```

```python
import time
import asyncio
import confirm.cloud.clickhouse as ch
import clickhouse_connect
con_details = ch.get_ch_config()
con_details['database'] = 'unnamed_7483655fa9c1418fa12f79915a9bf545'
ch_client = clickhouse_connect.get_client(**con_details, connect_timeout=60)
name = "results"
cols = ch.get_create_table_cols(df)
ch.command(
    ch_client,
    f"""
                CREATE TABLE IF NOT EXISTS {name} ({",".join(cols)})
                ENGINE = MergeTree()
                ORDER BY ()
                """,
    ch.default_insert_settings,
)

```

```python
start = time.time()
n = 10000
tasks = [
    asyncio.to_thread(
        ch_client.insert_arrow,
        "results",
        arrow_tbl[i * n : (i + 1) * n],
        settings=ch.async_insert_settings,
    )
    for i in range(64)
]
await asyncio.gather(*tasks)
print(time.time() - start)

```

```python
%%time
con.execute("copy (select * from results where step_id > 7 and step_id <= 10) to 's3://confirmasaurus-db-backup/results' (FORMAT parquet, PARTITION_BY ('step_id'), ALLOW_OVERWRITE TRUE)")
```

```python
df.shape
```

```python
ch_client.database
```

```python
import numpy as np
import pandas as pd
import pyarrow as pa
arr = np.random.uniform(size=(400000, 130))
df = pd.DataFrame(arr)
arrow_tbl = pa.Table.from_pandas(df)

import duckdb
import time
con = duckdb.connect()
con.execute("""
install httpfs;
load httpfs;
set s3_region='us-east-1';
set s3_access_key_id='';
set s3_secret_access_key='';
""")
start = time.time()
con.execute("copy (select * from arrow_tbl) to 's3://confirmasaurus-db-backup/results' (FORMAT parquet)")
print('took ', time.time() - start)
print('took ', time.time() - start)
print('took ', time.time() - start)
print('took ', time.time() - start)
print('took ', time.time() - start)
```

```python
import confirm.cloud.modal_util as modal_util
import modal
import numpy as np
import pandas as pd
import pyarrow as pa
modal_cfg = modal_util.get_defaults()

stub = modal.Stub()

@stub.function(**modal_cfg)
def test_fnc():
    arr = np.random.uniform(size=(400000, 130))
    df = pd.DataFrame(arr)
    arrow_tbl = pa.Table.from_pandas(df)

    import duckdb
    import time
    con = duckdb.connect()
    con.execute("""
    install httpfs;
    load httpfs;
    set s3_region='us-east-1';
    set s3_access_key_id='';
    set s3_secret_access_key='';
    """)
    start = time.time()
    con.execute("copy (select * from arrow_tbl) to 's3://confirmasaurus-db-backup/results' (FORMAT parquet)")
    print('took ', time.time() - start)
    print('took ', time.time() - start)
    print('took ', time.time() - start)
    print('took ', time.time() - start)
    print('took ', time.time() - start)
    
    # ch_client = ch.connect('unnamed_7483655fa9c1418fa12f79915a9bf545')
    # async def insert():
    #     n = 10000
    #     tasks = [
    #         asyncio.to_thread(
    #             ch_client.insert_arrow,
    #             "results",
    #             arrow_tbl[i * n : (i + 1) * n],
    #             settings=ch.async_insert_settings,
    #         )
    #         for i in range(64)
    #     ]
    #     await asyncio.gather(*tasks)
    # asyncio.run(insert())

with stub.run():
    test_fnc.call()
```
