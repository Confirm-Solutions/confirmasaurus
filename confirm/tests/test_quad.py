import numpy as np

import confirm.outlaw.quad as quad


def gen_f(scale):
    def f(x):
        return np.sin(x) ** 2 * np.exp(-((x / scale) ** 2))

    return f


def analytic_I(scale):
    return (
        0.5 * np.sqrt(np.pi) * np.exp(-(scale**2)) * (np.exp(scale**2) - 1) * scale
    )


def test_gauss_hermite():
    for scale in np.linspace(0.3, 4.0, 7):
        f = gen_f(scale)
        # The order here is excessive, but as scale increases, the oscillation
        # increases and makes the integral harder to compute.
        qr = quad.gauss_herm_rule(30, scale=scale)
        correct = analytic_I(scale)
        est = np.sum(f(qr.pts) * qr.wts)
        np.testing.assert_allclose(est, correct, rtol=1e-5)
