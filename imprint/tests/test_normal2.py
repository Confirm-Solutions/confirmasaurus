import numpy as np

import imprint.bound.normal2 as normal2


def test_A_secant_unrestricted_no_nan():
    n = np.array([10, 10])
    theta1 = np.array([5, 2])
    theta2 = np.array([-0.1, -0.2])
    v1 = np.array([-0.2, 0.1])
    v2 = np.array([-0.1, -10])

    out = normal2.A_secant(n, theta1, theta2, v1, v2, q=1)
    assert not np.isnan(out)

    out = normal2.A_secant(n, theta1, theta2, v1, v2, q=5)
    assert not np.isnan(out)

    out = normal2.A_secant(n, theta1, theta2, v1, v2, q=1e30)
    print(out)
    assert not np.isnan(out)
