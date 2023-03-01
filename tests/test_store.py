import time

import pandas as pd
import pytest

from confirm.adagrid.store import DuckDBStore
from confirm.adagrid.store import is_table_name
from confirm.adagrid.store import PandasStore

ex = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]))
ex2 = ex.copy()
ex2["a"] += 1


def test_is_table_name():
    assert is_table_name("hello")
    assert is_table_name("hello_world")
    assert is_table_name("lol0")
    assert not is_table_name("123_hello")
    assert not is_table_name("_hello")
    assert not is_table_name("hello!")


class StoreTester:
    def test_exists_not_exists(self):
        assert not self.connect().exists("key")

    def test_set_get_df(self):
        c = self.connect()
        c.set("key", ex)
        assert c.exists("key")
        c.set("key", ex)
        pd.testing.assert_frame_equal(c.get("key"), ex)

    def test_overwrite(self):
        c = self.connect()
        c.set("key", ex)
        mod = ex.drop(columns=["b"])
        c.set("key", mod)
        pd.testing.assert_frame_equal(c.get("key"), mod)

    def test_append(self):
        c = self.connect()
        c.set("key", ex)
        pd.testing.assert_frame_equal(c.get("key"), ex)
        c.set_or_append("key", ex2)
        # NOTE: it's possible for the first insert to happen *after* the second
        # insert because of Clickhouse's very weak consistency guarantees.
        # Currently, we tolerate this. This test accomodates this rare
        # possibility by sorting the data.
        pd.testing.assert_frame_equal(
            c.get("key").sort_values(by=["a", "b"]).reset_index(drop=True),
            pd.concat([ex, ex2], axis=0)
            .sort_values(by=["a", "b"])
            .reset_index(drop=True),
        )

    def test_get_not_exists(self):
        c = self.connect()
        with pytest.raises(KeyError):
            c.get("key")

    def test_append_not_exists(self):
        c = self.connect()
        c.set_or_append("key", ex)
        pd.testing.assert_frame_equal(c.get("key"), ex)

    def test_cached_function(self):
        c = self.connect()

        def f(x):
            return pd.DataFrame(dict(x=[time.time() + x]))

        r1 = c(f)(1)
        r2 = c(f)(1)
        assert r1.iloc[0, 0] == r2.iloc[0, 0]

        r3 = c(f)(2)
        assert r1.iloc[0, 0] != r3.iloc[0, 0]


class TestPandasStore(StoreTester):
    def connect(self):
        return PandasStore()


class TestDuckDBStore(StoreTester):
    def connect(self):
        return DuckDBStore.connect()

    def test_set_nickname(self):
        c = self.connect()
        c.set("0key", ex, nickname="hi")
        pd.testing.assert_frame_equal(
            c.con.execute("select * from _store_hi_0").df(), ex
        )

    def test_set_legal_table_name(self):
        c = self.connect()
        c.set("key", ex)
        pd.testing.assert_frame_equal(c.con.execute("select * from key").df(), ex)
