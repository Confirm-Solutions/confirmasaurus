```python
import os
import keyring
import redis

host = keyring.get_password("upstash-confirm-coordinator-host", os.environ["USER"])
password = keyring.get_password(
    "upstash-confirm-coordinator-password", os.environ["USER"]
)

r = redis.Redis(host=host, port="37085", password=password)
r.set("foo", "bar")
```

```python
%%time
print(r.get("foo"))
```
