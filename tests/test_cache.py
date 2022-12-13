import time

import pandas as pd
import pytest

from confirm.imprint.cache import DuckDBCache
from confirm.imprint.cache import PandasCache

ex = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]))


class CacheTester:
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
        c.append("key", ex)
        pd.testing.assert_frame_equal(
            c.get("key"), pd.concat([ex, ex], axis=0).reset_index(drop=True)
        )

    def test_get_not_exists(self):
        c = self.connect()
        with pytest.raises(KeyError):
            c.get("key")

    def test_append_not_exists(self):
        c = self.connect()
        with pytest.raises(KeyError):
            c.append("key", ex)

    def test_cached_function(self):
        c = self.connect()

        def f(x):
            return pd.DataFrame(dict(x=[time.time() + x]))

        r1 = c(f)(1)
        r2 = c(f)(1)
        assert r1.iloc[0, 0] == r2.iloc[0, 0]

        r3 = c(f)(2)
        assert r1.iloc[0, 0] != r3.iloc[0, 0]


class TestPandasCache(CacheTester):
    def connect(self):
        return PandasCache()


class TestDuckDBCache(CacheTester):
    def connect(self):
        return DuckDBCache.connect()

    def test_set_nickname(self):
        c = self.connect()
        c.set("key", ex, nickname="hi")
        pd.testing.assert_frame_equal(
            c.con.execute("select * from _cache_hi_0").df(), ex
        )
