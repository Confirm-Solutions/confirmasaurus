import json
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

import duckdb
import pandas as pd


class Cache:
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
class PandasCache(Cache):
    """
    A very simple reference cache implementation that stores DataFrames in memory.
    """

    _store: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, key):
        return self._store[key]

    def set(self, key, data, nickname=None):
        self._store[key] = data

    def append(self, key, data):
        self._store[key] = pd.concat([self._store[key], data], axis=0).reset_index(
            drop=True
        )

    def exists(self, key):
        return key in self._store


@dataclass
class DuckDBCache(Cache):
    con: duckdb.DuckDBPyConnection

    def __post_init__(self):
        self.con.execute(
            "create table if not exists cache_tables (key VARCHAR, table_name VARCHAR)"
        )

    def _get_all_keys(self):
        return self.con.execute("select * from cache_tables").df()

    def get(self, key):
        exists, table_name = self._exists(key)
        if exists:
            return self.con.execute(f"select * from {table_name}").df()
        else:
            raise KeyError(f"Key {key} not found in cache")

    def set(self, key, df, nickname=None):
        exists, table_name = self._exists(key)
        if exists:
            self.con.execute(f"drop table {table_name}")
        else:
            idx = self.con.execute("select count(*) from cache_tables").fetchall()[0][0]
            table_name = (
                f"_cache_{nickname}_{idx}" if nickname is not None else f"_cache_{idx}"
            )
            self.con.execute(
                "insert into cache_tables values (?, ?)", (key, table_name)
            )
        self.con.execute(f"create table {table_name} as select * from df ")

    def append(self, key, data):
        exists, table_name = self._exists(key)
        if exists:
            self.con.execute(f"insert into {table_name} select * from data")
        else:
            raise KeyError(f"Key {key} not found in cache")

    def _exists(self, key):
        table_name = self.con.execute(
            "select table_name from cache_tables where key = ?", (key,)
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
        Load a cache database from a file.

        Args:
            path: The filepath to the database.

        Returns:
            The database.
        """
        return DuckDBCache(duckdb.connect(path))
