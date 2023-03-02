import json
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import duckdb


def is_table_name(s):
    if not s:
        return False

    if not s[0].isalpha():
        return False

    if not re.match("^[a-zA-Z0-9_]+$", s):
        return False

    return True


class Store:
    """
    A Store is a key-value store that can save and retrieve dataframes in a
    database. We use this for two purposes:
    1. To cache the results of expensive computations.
    2. To store the configuration of a job in the database.

    Tile-related information is handled separately. For example, by DuckDBTiles.
    """

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            key = json.dumps(
                dict(
                    f=func.__module__ + "." + func.__qualname__,
                    args=args,
                    kwargs={str(k): str(v) for k, v in kwargs.items()},
                )
            )
            if self.exists(key):
                return self.get(key)
            else:
                result = func(*args, **kwargs)
                self.set(key, result, nickname=func.__name__)
                return result

        return wrapper


@dataclass
class PandasStore(Store):
    """
    A very simple reference store implementation that stores DataFrames in memory.
    """

    _store: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, key):
        return self._store[key]

    def set(self, key, data, nickname=None):
        self._store[key] = data

    def set_or_append(self, key, data):
        if self.exists(key):
            self._store[key] = pd.concat([self._store[key], data], axis=0).reset_index(
                drop=True
            )
        else:
            self.set(key, data)

    def exists(self, key):
        return key in self._store


@dataclass
class DuckDBStore(Store):
    con: "duckdb.DuckDBPyConnection"

    def __post_init__(self):
        self.con.execute(
            "create table if not exists store_tables (key VARCHAR, table_name VARCHAR)"
        )

    def _get_all_keys(self):
        return self.con.execute("select * from store_tables").df()

    def get(self, key):
        exists, table_name = self._exists(key)
        if exists:
            return self.con.execute(f"select * from {table_name}").df()
        else:
            raise KeyError(f"Key {key} not found in store")

    def set(self, key, df, nickname=None):
        """
        Set a key. If the key already exists, it will be overwritten.

        If the key is a valid table name, the key will be stored in that table.
        If the key is not a valid table name:
        - and nickname is not None, a new table will be created with a name
          of the form _store_{nickname}_{idx}.
        - and nickname is None, a new table will be created with a name of the
          form _store_{idx}.

        Args:
            key: The key
            df: The
            nickname: _description_. Defaults to None.
        """
        exists, table_name = self._exists(key)
        if exists:
            self.con.execute(f"drop table {table_name}")
        else:
            q = self.con.execute("select count(*) from store_tables").fetchone()
            idx = 0 if q is None else q[0]
            if is_table_name(key):
                table_name = key
            else:
                table_name = (
                    f"_store_{nickname}_{idx}"
                    if nickname is not None
                    else f"_store_{idx}"
                )
            self.con.execute(
                "insert into store_tables values (?, ?)", (key, table_name)
            )
        self.con.execute(f"create table {table_name} as select * from df")

    def set_or_append(self, key, df):
        exists, table_name = self._exists(key)
        if exists:
            self.con.execute(f"insert into {table_name} select * from df")
        else:
            self.set(key, df)

    def _exists(self, key):
        table_name = self.con.execute(
            "select table_name from store_tables where key = ?", (key,)
        ).fetchall()
        if len(table_name) == 0:
            return False, None
        else:
            return True, table_name[0][0]

    def exists(self, key):
        return self._exists(key)[0]

    @staticmethod
    def connect(path=":memory:"):
        """
        Load a store database from a file.

        Args:
            path: The filepath to the database.

        Returns:
            The database.
        """
        import duckdb

        return DuckDBStore(duckdb.connect(path))
