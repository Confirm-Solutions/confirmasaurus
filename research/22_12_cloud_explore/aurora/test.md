```python
import pandas as pd

full_dfdf = pd.read_parquet("./dbtest.parquet")
```

```python
full_df = df
df = full_df.iloc[:40000]
```

```python
import awswrangler as wr
```

```python
wr.data_api.rds.connect(
    "imprint-aurora-instance-1.canndqdikygz.us-east-1.rds.amazonaws.com", "postgres", ""
)
```

```python
import psycopg2

con = psycopg2.connect(
    host="imprint-aurora-instance-1.canndqdikygz.us-east-1.rds.amazonaws.com",
    user="postgres",
    password="",
    dbname="postgres",
)
```

```python
query = "create table tiles (" + ",".join([f"{c} real" for c in df.columns]) + ")"
query
```

```python
con.commit()
```

```python
cur = con.cursor()
cur.execute(query)
con.commit()
```

```python
from sqlalchemy import create_engine
import psycopg2
import io

engine = create_engine(
    "postgresql+psycopg2://postgres:password@imprint-aurora-instance-1.canndqdikygz.us-east-1.rds.amazonaws.com:5432/postgres"
)

df.head(0).to_sql(
    "table_name", engine, if_exists="replace", index=False
)  # drops old table and creates new empty table
```

```python
conn = engine.raw_connection()
cur = conn.cursor()
output = io.StringIO()
df.to_csv(output, sep="\t", header=False, index=False)
output.seek(0)
contents = output.getvalue()
cur.copy_from(output, "table_name", null="")  # null values become ''
conn.commit()
```

```python
df_read = pd.read_sql(
    "select * from table_name order by orig_lam limit 10000", con=engine
)
```

```python
pd.read_sql("select count(*) from table_name", con=engine)
```

```python
df_read
```
