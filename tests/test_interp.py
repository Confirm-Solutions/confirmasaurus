import jax
import jax.numpy as jnp
import numpy as np
import pytest

from confirm.outlaw.interp import interpn


def test_interpn():
    grid = jnp.array([[0, 1], [0, 1]])
    values = jnp.array([[0, 1], [2, 3]])
    xi = jnp.array([[0.5, 0.5], [1.0, 0.0]])
    result = jax.vmap(interpn, in_axes=(None, None, 0))(grid, values, xi)
    np.testing.assert_allclose(result, [1.5, 2.0])


@pytest.mark.parametrize("dim", [1, 3])
def test_against_scipy_multi_value(dim):
    import scipy.interpolate

    for i in range(3):
        np.random.seed(10)
        grid = [np.sort(np.random.uniform(size=10)) for _ in range(2)]
        values = jnp.array(np.random.uniform(size=(10, 10, dim)).squeeze())
        xi = np.random.uniform(size=(10, 2))
        result = jax.vmap(interpn, in_axes=(None, None, 0))(grid, values, xi)
        scipy_result = scipy.interpolate.interpn(
            grid,
            np.asarray(values).copy(),
            xi,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        np.testing.assert_allclose(result, scipy_result)
