import numpy as np

from confirm.lewislib.batch import batch
from confirm.lewislib.batch import batch_yield


def test_simple():
    def f(x):
        return x + 1

    batched_f = batch_yield(f, batch_size=2, in_axes=(0,))
    out = list(batched_f(np.array([1, 2, 3, 4])))
    assert len(out) == 2
    np.testing.assert_allclose(out[0][0], np.array([2, 3]))
    assert out[0][1] == 0
    np.testing.assert_allclose(out[1][0], np.array([4, 5]))
    assert out[1][1] == 0


def test_pad():
    def f(x):
        return x + 1

    batched_f = batch_yield(f, batch_size=3, in_axes=(0,))
    out = list(batched_f(np.array([1, 2, 3, 4])))
    np.testing.assert_allclose(out[1][0], np.array([5, 5, 5]))
    assert out[1][1] == 2


def test_pass_interval():
    def f(s, e, x):
        return x[s:e] + 1

    batched_f = batch_yield(f, batch_size=3, in_axes=(None,), pass_interval=True)
    out = list(batched_f(np.array([1, 2, 3, 4])))
    np.testing.assert_allclose(out[1][0], np.array([5, 5, 5]))
    assert out[1][1] == 2


def test_multidim():
    def f(x):
        return (x.sum(axis=1), x.prod(axis=1))

    for d in range(1, 15):
        inputs = np.random.rand(d, 5)
        batched_f = batch(f, batch_size=5, in_axes=(0,))
        out = batched_f(inputs)
        np.testing.assert_allclose(out[0], inputs.sum(axis=1))
        np.testing.assert_allclose(out[1], inputs.prod(axis=1))


def test_multidim_single():
    def f(x):
        return x.sum(axis=1)

    inputs = np.random.rand(7, 5)
    batched_f = batch(f, batch_size=5, in_axes=(0,))
    out = batched_f(inputs)
    np.testing.assert_allclose(out, inputs.sum(axis=1))


def test_out_axes1():
    def f(x):
        return x.T

    inputs = np.random.rand(7, 5)
    batched_f = batch(f, batch_size=5, in_axes=(0,), out_axes=(1,))
    out = batched_f(inputs)
    np.testing.assert_allclose(out, inputs.T)
