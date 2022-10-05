import numpy as np

from confirm.lewislib.batch import batch


def test_batch_simple():
    def f(x):
        return x + 1

    batched_f = batch(f, batch_size=2, in_axes=(0,))
    out = list(batched_f(np.array([1, 2, 3, 4])))
    np.testing.assert_allclose(out[0][0], np.array([2, 3]))
    assert out[0][1] == 0
    np.testing.assert_allclose(out[1][0], np.array([4, 5]))
    assert out[1][1] == 0


def test_batch_pad():
    def f(x):
        return x + 1

    batched_f = batch(f, batch_size=3, in_axes=(0,))
    out = list(batched_f(np.array([1, 2, 3, 4])))
    np.testing.assert_allclose(out[1][0], np.array([5, 5, 5]))
    assert out[1][1] == 2
