```python
import logging
import asyncio
import contextlib
from imprint.nb_util import setup_nb
setup_nb()
from confirm.adagrid.backend import backup_daemon

    
class DB:
    calls = 0
    def backup(self):
        self.calls += 1

db = DB()
async with backup_daemon(db, backup_interval=1):
    print("Hello world")
    await asyncio.sleep(3)
assert(db.calls == 3)
```

```python

```
