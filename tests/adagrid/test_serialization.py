import numpy as np

import imprint as ip
from confirm.adagrid.init import _deserialize_null_hypos
from confirm.adagrid.init import _serialize_null_hypos


def test_planar_null():
    hypos = [ip.hypo("x1 < 0"), ip.hypo("x0 < 1")]
    null_hypos_df = _serialize_null_hypos(hypos)
    hypos2 = _deserialize_null_hypos(null_hypos_df)
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


def test_custom_null():
    hypos = [CustomNull("hi")]
    null_hypos_df = _serialize_null_hypos(hypos)
    hypos2 = _deserialize_null_hypos(null_hypos_df)
    assert hypos2[0].name == "hi"
    np.testing.assert_allclose(hypos2[0].dist(np.array([[3, 4]])), 5)
