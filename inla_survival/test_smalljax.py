import timeit
import pytest
import jax
import numpy as np
from smalljax import inv22, inv33, inv44, inv_recurse, logdet, gen, invJI


@pytest.mark.parametrize(
    "fnc, n",
    [
        (inv22, 2),
        (inv33, 3),
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
        lu = gen(f'lu{d}')(m)
        print(lu)
        L = np.tril(lu, k=-1)
        print(L)
        np.fill_diagonal(L, np.full(d, 1.0))
        U = np.triu(lu)
        np.testing.assert_allclose(L.dot(U), m, atol=1e-7)
        
def test_solve():
    np.random.seed(10)
    for d in range(2, 5):
        a = np.random.rand(d, d)
        b = np.random.rand(d)
        x = gen(f'solve{d}')(a, b)
        np.testing.assert_allclose(x, np.linalg.solve(a, b), atol=1e-7)

def test_logdet():
    np.random.seed(10)
    for d in range(2, 5):
        a = np.random.rand(d, d)
        sign, correct = np.linalg.slogdet(a)
        v = logdet(a)
        np.testing.assert_allclose(v, correct, atol=1e-7)
        

def test_invJI():
    np.random.seed(10)
    a, b = np.random.rand(2)
    for d in range(2, 10):
        m = np.full((d,d), b) + np.diag(np.full(d, a - b))
        correct = np.linalg.inv(m)
        ainv, binv = invJI(a, b, d)
        m = np.full((d,d), binv) + np.diag(np.full(d, ainv - binv))
        np.testing.assert_allclose(m, correct, rtol=1e-6)
        
def my_timeit(f):
    return np.min(timeit.repeat(f, number = 100))

def benchmark_invJI():
    for d in range(2, 10):
        N = int(300000 / (d ** 2))
        vs = np.random.rand(N, 2)
        a = vs[:,0]
        b = vs[:,1]
        m = b[:, None, None] * np.full((d,d), 1.0) + (a - b)[:, None, None] * np.eye(d)
        vmap_inv_recurse = jax.jit(jax.vmap(inv_recurse))
        vmap_invJI = jax.jit(jax.vmap(invJI, in_axes=(0,0,None)))
        print('\n', d)
        print(f'inv_recurse', my_timeit(lambda: vmap_inv_recurse(m)))
        print('invJI', my_timeit(lambda: vmap_invJI(a, b, d)))
    
    
if __name__ == "__main__":
    benchmark_invJI()