import jax
import numpy as np
import pytest

from inlaw.smalljax import gen
from inlaw.smalljax import inv22
from inlaw.smalljax import inv33
from inlaw.smalljax import inv44
from inlaw.smalljax import inv_recurse
from inlaw.smalljax import logdet


@pytest.mark.parametrize(
    "fnc, n",
    [
        (inv22, 2),
        (inv33, 3),
        (inv44, 4),
        (inv_recurse, 2),
        (inv_recurse, 3),
        (inv_recurse, 4),
    ],
)
def test_inv(fnc, n):
    np.random.seed(10)
    vmap_f = jax.jit(jax.vmap(fnc))
    mats = np.random.rand(10, n, n)
    inv = vmap_f(mats)
    correct = np.linalg.inv(mats)
    np.testing.assert_allclose(inv, correct, atol=1e-4, rtol=1e-4)


def test_lu():
    np.random.seed(10)
    for d in range(2, 5):
        m = np.random.rand(d, d)
        lu = gen(f"lu{d}")(m)
        L = np.tril(lu, k=-1)
        np.fill_diagonal(L, np.full(d, 1.0))
        U = np.triu(lu)
        np.testing.assert_allclose(L.dot(U), m, atol=1e-7)


def test_solve():
    np.random.seed(10)
    for d in range(2, 5):
        a = np.random.rand(d, d)
        b = np.random.rand(d)
        x = gen(f"solve{d}")(a, b)
        np.testing.assert_allclose(x, np.linalg.solve(a, b), atol=1e-7)


def test_logdet():
    np.random.seed(10)
    for d in range(2, 5):
        a = np.random.rand(d, d)
        sign, correct = np.linalg.slogdet(a)
        v = logdet(a)
        np.testing.assert_allclose(v, correct, atol=1e-7)
