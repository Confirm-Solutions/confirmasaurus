import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def bench(f, N, d, iters=5):
    np.random.seed(10)
    mats = np.random.rand(int(N), d, d)
    jax_mats = jnp.array(mats)
    for i in range(iters):
        start = time.time()
        # correct = jnp.linalg.inv(jax_mats).block_until_ready()
        _ = jnp.linalg.inv(jax_mats).block_until_ready()
        end = time.time()
    jli_time = end - start
    for i in range(iters):
        start = time.time()
        # fout = f(jax_mats).block_until_ready()
        _ = f(jax_mats).block_until_ready()
        end = time.time()
    f_time = end - start
    # np.testing.assert_allclose(fout, correct, rtol=1e-4)
    return jli_time / N * 1e6, f_time / N * 1e6


def inv22(mat):
    m1, m2 = mat[0]
    m3, m4 = mat[1]
    inv_det = 1.0 / (m1 * m4 - m2 * m3)
    return jnp.array([[m4, -m2], [-m3, m1]]) * inv_det


vmap_inv22 = jax.jit(jax.vmap(inv22))


def inv33(mat):
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    inv_det = 1.0 / det
    return (
        jnp.array(
            [
                [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
                [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
                [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
            ]
        )
        * inv_det
    )


vmap_inv33 = jax.jit(jax.vmap(inv33))


def inv44(m):
    """
    See https://github.com/willnode/N-Matrix-Programmer
    for source in the "Info" folder
    MIT License.
    """
    A2323 = m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]
    A1323 = m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]
    A1223 = m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]
    A0323 = m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]
    A0223 = m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]
    A0123 = m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0]
    A2313 = m[1, 2] * m[3, 3] - m[1, 3] * m[3, 2]
    A1313 = m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1]
    A1213 = m[1, 1] * m[3, 2] - m[1, 2] * m[3, 1]
    A2312 = m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2]
    A1312 = m[1, 1] * m[2, 3] - m[1, 3] * m[2, 1]
    A1212 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    A0313 = m[1, 0] * m[3, 3] - m[1, 3] * m[3, 0]
    A0213 = m[1, 0] * m[3, 2] - m[1, 2] * m[3, 0]
    A0312 = m[1, 0] * m[2, 3] - m[1, 3] * m[2, 0]
    A0212 = m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]
    A0113 = m[1, 0] * m[3, 1] - m[1, 1] * m[3, 0]
    A0112 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]

    det = (
        m[0, 0] * (m[1, 1] * A2323 - m[1, 2] * A1323 + m[1, 3] * A1223)
        - m[0, 1] * (m[1, 0] * A2323 - m[1, 2] * A0323 + m[1, 3] * A0223)
        + m[0, 2] * (m[1, 0] * A1323 - m[1, 1] * A0323 + m[1, 3] * A0123)
        - m[0, 3] * (m[1, 0] * A1223 - m[1, 1] * A0223 + m[1, 2] * A0123)
    )
    invdet = 1.0 / det

    return invdet * jnp.array(
        [
            (m[1, 1] * A2323 - m[1, 2] * A1323 + m[1, 3] * A1223),
            -(m[0, 1] * A2323 - m[0, 2] * A1323 + m[0, 3] * A1223),
            (m[0, 1] * A2313 - m[0, 2] * A1313 + m[0, 3] * A1213),
            -(m[0, 1] * A2312 - m[0, 2] * A1312 + m[0, 3] * A1212),
            -(m[1, 0] * A2323 - m[1, 2] * A0323 + m[1, 3] * A0223),
            (m[0, 0] * A2323 - m[0, 2] * A0323 + m[0, 3] * A0223),
            -(m[0, 0] * A2313 - m[0, 2] * A0313 + m[0, 3] * A0213),
            (m[0, 0] * A2312 - m[0, 2] * A0312 + m[0, 3] * A0212),
            (m[1, 0] * A1323 - m[1, 1] * A0323 + m[1, 3] * A0123),
            -(m[0, 0] * A1323 - m[0, 1] * A0323 + m[0, 3] * A0123),
            (m[0, 0] * A1313 - m[0, 1] * A0313 + m[0, 3] * A0113),
            -(m[0, 0] * A1312 - m[0, 1] * A0312 + m[0, 3] * A0112),
            -(m[1, 0] * A1223 - m[1, 1] * A0223 + m[1, 2] * A0123),
            (m[0, 0] * A1223 - m[0, 1] * A0223 + m[0, 2] * A0123),
            -(m[0, 0] * A1213 - m[0, 1] * A0213 + m[0, 2] * A0113),
            (m[0, 0] * A1212 - m[0, 1] * A0212 + m[0, 2] * A0112),
        ]
    ).reshape((4, 4))


vmap_inv44 = jax.jit(jax.vmap(inv44))


def fast_dot(a, b):
    return (a * b).sum()


fast_mat_mul = jax.vmap(jax.vmap(fast_dot, in_axes=(None, 1)), in_axes=(0, None))


def inv_recurse(mat):
    if mat.shape[0] == 1:
        return 1.0 / mat
    if mat.shape[0] == 2:
        return inv22(mat)
    elif mat.shape[0] == 3:
        return inv33(mat)
    elif mat.shape[0] == 4:
        return inv44(mat)
    r = 4
    A = mat[:r, :r]
    B = mat[:r, r:]
    C = mat[r:, :r]
    D = mat[r:, r:]
    A_inv = inv_recurse(A)
    CA_inv = fast_mat_mul(C, A_inv)
    schur = D - fast_mat_mul(CA_inv, B)
    schur_inv = inv_recurse(schur)
    A_invB = fast_mat_mul(A_inv, B)
    lr = schur_inv
    ur = -fast_mat_mul(A_invB, schur_inv)
    ll = -fast_mat_mul(schur_inv, CA_inv)
    ul = A_inv - fast_mat_mul(A_invB, ll)
    return jnp.concatenate(
        (
            jnp.concatenate((ul, ur), axis=1),
            jnp.concatenate((ll, lr), axis=1),
        ),
        axis=0,
    )


vmap_inv_recurse = jax.jit(jax.vmap(inv_recurse))


@pytest.mark.parametrize(
    "fnc, n",
    [
        (vmap_inv22, 2),
        (vmap_inv33, 3),
        (vmap_inv_recurse, 2),
        (vmap_inv_recurse, 3),
        (vmap_inv_recurse, 4),
    ],
)
def test_inv(fnc, n):
    np.random.seed(10)
    mats = np.random.rand(10, n, n)
    inv = fnc(mats)
    correct = np.linalg.inv(mats)
    np.testing.assert_allclose(inv, correct, atol=1e-4, rtol=1e-4)
