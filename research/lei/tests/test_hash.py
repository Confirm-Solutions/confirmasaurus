import jax
import jax.numpy as jnp
import numpy as np
from lewis.lookup_table import LookupTable


default_params = {
    "n_arms": 3,
    "n_stage_1": 3,
    "n_stage_2": 3,
    "n_stage_1_interims": 1,
    "n_stage_1_add_per_interim": 10,
    "n_stage_2_add_per_interim": 4,
    "stage_1_futility_threshold": 0.1,
    "stage_2_futility_threshold": 0.1,
    "stage_1_efficacy_threshold": 0.1,
    "stage_2_efficacy_threshold": 0.9,
    "inter_stage_futility_threshold": 0.8,
    "posterior_difference_threshold": 0.05,
    "rejection_threshold": 0.05,
}


def test_stable_sort_1():
    n = jnp.array([20, 20, 10, 20])
    order = jnp.flip(n.shape[0] - 1 - jnp.argsort(jnp.flip(n), kind="stable"))
    expected = jnp.array([0, 1, 3, 2])
    assert jnp.array_equal(order, expected)


def test_stable_sort_2():
    n = jnp.array([30, 10, 20, 30])
    order = jnp.flip(n.shape[0] - 1 - jnp.argsort(jnp.flip(n), kind="stable"))
    expected = jnp.array([0, 3, 2, 1])
    assert jnp.array_equal(order, expected)


def test_hash_undo():
    n = jnp.array([12, 5, 5, 12])
    n_order = jnp.flip(n.shape[0] - 1 - jnp.argsort(jnp.flip(n), kind="stable"))

    # test if this piece of code gives us the correct undoing
    actual = jnp.argsort(n_order)
    expected = jnp.array([0, 2, 3, 1])

    assert jnp.array_equal(actual, expected)


def test_y_to_index():
    n = np.array([5, 5, 5, 2])
    max_idx = np.prod(n + 1)
    y = np.zeros(n.shape[0])

    def increment(y):
        carry = 1
        for i in range(n.shape[-1] - 1, -1, -1):
            y[i] += carry
            carry = y[i] // (n[i] + 1)
            y[i] -= carry * (n[i] + 1)
        return y

    for i in range(max_idx):
        actual = y[-1] + jnp.sum(jnp.flip(y[:-1]) * jnp.cumprod(jnp.flip(n[1:] + 1)))
        expected = i
        assert jnp.array_equal(actual, expected)
        y = increment(y)


def test_hash():
    n = jnp.array([12, 5, 5, 12])
    dims = n + 1
    values = jnp.arange(0, jnp.prod(dims))[:, None]
    table = LookupTable(dims[None], values)
    y = jnp.array([5, 1, 0, 10])

    # tests if the at function is jit-able also.
    @jax.jit
    def internal():
        data = jnp.stack((y, dims), axis=-1)
        return table.at(data)[0].squeeze()

    actual = internal()
    expected = 2428
    assert jnp.array_equal(actual, expected)
