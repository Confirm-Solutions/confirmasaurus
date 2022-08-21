import numpy as np
import scipy.interpolate

from outlaw.interp import interpn


def test_interpn():
    grid = np.array([[0, 1], [0, 1]])
    values = np.array([[0, 1], [2, 3]])
    xi = np.array([[0.5, 0.5], [1.0, 0.0]])
    result = interpn(grid, values, xi)
    np.testing.assert_allclose(result, [1.5, 2.0])


def test_against_scipy():
    for i in range(3):
        np.random.seed(10)
        grid = [np.sort(np.random.uniform(size=10)) for _ in range(2)]
        values = np.random.uniform(size=(10, 10))
        xi = np.random.uniform(size=(10, 2))
        result = interpn(grid, values, xi)
        scipy_result = scipy.interpolate.interpn(
            grid, values, xi, method="linear", bounds_error=False, fill_value=None
        )
        np.testing.assert_allclose(result, scipy_result)
