import numpy as np
import pytest

import imprint as ip
from confirm.adagrid.adagrid import _load_null_hypos
from confirm.adagrid.adagrid import _store_null_hypos
from confirm.adagrid.db import DuckDBTiles


@pytest.fixture
def duckdb():
    return DuckDBTiles.connect()


@pytest.fixture(params=["duckdb", "ch_db"])
def db(request):
    return request.getfixturevalue(request.param)


def test_planar_null(db):
    hypos = [ip.hypo("x1 < 0"), ip.hypo("x0 < 1")]
    _store_null_hypos(db, hypos)
    hypos2 = _load_null_hypos(db)
    for i in range(2):
        np.testing.assert_allclose(hypos[i].c, hypos2[i].c)
        np.testing.assert_allclose(hypos[i].n, hypos2[i].n)


class CustomNull(ip.NullHypothesis):
    def __init__(self, name):
        self.name = name

    def dist(self, theta):
        return np.linalg.norm(theta, axis=1)

    def description(self):
        return "CustomNull"


def test_custom_null(db):
    hypos = [CustomNull("hi")]
    _store_null_hypos(db, hypos)
    hypos2 = _load_null_hypos(db)
    assert hypos2[0].name == "hi"
    np.testing.assert_allclose(hypos2[0].dist(np.array([[3, 4]])), 5)
