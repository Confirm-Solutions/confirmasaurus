```python
import confirm.outlaw.nb_util as nb_util
nb_util.setup_nb()
import xxhash

id = [0, 1]
```

```python
import confirm.cloud.clickhouse as ch
import pandas as pd
import numpy as np
# ch_db = ch.Clickhouse.connect()
```

```python
df = pd.read_parquet('research/cloud_explore/clickhouse/dbtest.parquet')
```

```python
import string
import random
# ids = [
#     ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(80))
#     for i in range(100000)
# ]
A, Z = np.array(["A","Z"]).view("int32") 

NO_CODES = df.shape[0]
LEN = 80

ids = np.random.randint(low=A,high=Z,size=NO_CODES*LEN,dtype="int32").view(f"U{LEN}")
```

```python
# ids_hashed = [xxhash.xxh128_digest(ids[i]) for i in range(ids.shape[0])]
```

```python
ch_db = ch.Clickhouse.connect(host='localhost', port=8123, password='')
```

```python
df['id'] = df.index
# df['id'] = ids
# df['id'] = df['id'].astype('|S80')
# df['id'] = ids_hashed
# df['id'] = df['id'].astype("|S16")
df['eligible'] = True
df['active'] = True
df['lineage'] = ids
df['lineage'] = df['id'].astype('|S80')
df['orderer'] = df['orig_lam']
```

```python
df.memory_usage().sum() / 1e9
```

```python
%%time
# ch_db.client.command('drop table tiles')
# ch_db.client.command('drop table tiles_inactive')
# ch_db.client.command('drop table work')
ch_db.init_tiles(df[:5000000])
```

```python
%%time
ch_db.write(df[5000000:10000000])
ch_db.write(df[10000000:15000000])
ch_db.write(df[15000000:20000000])
```

```python
%%time
ch_db.write(df[5000000:10000000])
ch_db.write(df[10000000:15000000])
ch_db.write(df[15000000:20000000])
```

```python
import time
runtimes = []
for i in range(5):
    start = time.time()
    ch_db.next(10000, 'orderer')
    print(i, time.time() - start)
    runtimes.append(time.time() - start)
```

```python
import confirm.imprint as ip
```

```python
%%time
ip.grid.gen_short_uuids(10000)
```

```python
%%time
for i in range(10000):
    xxhash.xxh64_intdigest(str(id))
```
