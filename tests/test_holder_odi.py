from itertools import product

import numpy as np
import pytest
from scipy.special import expit
from scipy.special import logit

import confirm.mini_imprint.binomial as binomial


def C6(n, p):
    return (
        n
        * p
        * (1 - p)
        * (
            1
            - 30 * p * (1 - p) * (1 - 4 * p * (1 - p))
            + 5 * n * p * (1 - p) * (5 - 26 * p * (1 - p))
            + 15 * n**2 * p**2 * (1 - p) ** 2
        )
    )


def test_odi_constant():
    """Test holderq=6 against the formula from wikipedia for a variety of (n, p)j"""
    C_f = binomial._build_odi_constant_func(6)
    C_fn = binomial._build_odi_constant_func_numerical(6)
    np.random.seed(0)
    ntest = 5
    n = 51
    p = np.random.uniform(0, 1, ntest)
    np.testing.assert_allclose(C_f(n, p), C6(n, p), rtol=1e-5)
    np.testing.assert_allclose(C_fn(n, p), C6(n, p), rtol=1e-5)


def test_calc_cqpp_05_crossing():
    """Test the Cqpp against a manual calculation including p=0.5 crossing."""
    theta_tiles = np.array([[-0.5, -0.5], [0.1, 0.05]])
    R = 0.09
    radii = np.full((2, 2), R)
    hypercube = np.array(list(product((1, -1), repeat=2)))
    tile_corners = theta_tiles[:, None, :] + hypercube[None, :] * radii[:, None]

    C_f = binomial._build_odi_constant_func(6)
    Cqpp = binomial._calc_Cqpp(theta_tiles, tile_corners, 50, 6, C_f)

    sup_v = (2 * (R**1.2)) ** (1.0 / 1.2)
    sup_moment0 = (C_f(50, expit(-0.41)) + C_f(50, expit(-0.41))) ** (1.0 / 6.0)
    sup_moment1 = (C_f(50, expit(0.01)) + C_f(50, expit(0.0))) ** (1.0 / 6.0)
    np.testing.assert_allclose(Cqpp, [sup_v * sup_moment0, sup_v * sup_moment1])


@pytest.mark.parametrize("d", [1, 2])
def test_holder_odi(d):
    """
    Test for a full end-to-end bound calculation. The "correct" outputs here
    are just fixed values from the current output.
    """
    n_arm_samples = 50
    nsims = 10000

    center_p = 0.2
    theta_tiles = np.full((1, d), logit(center_p))
    radii = np.array([-1.1 - theta_tiles[0, 0]])
    hypercube = np.array(list(product((1, -1), repeat=d)))
    tile_corners = theta_tiles + hypercube[None, :] * radii
    is_null_per_arm = np.full_like(theta_tiles, 1, dtype=bool)

    delta = 0.01
    holderq = 6
    thresh = 20

    np.random.seed(17)
    uniforms = np.random.uniform(size=(nsims, n_arm_samples, d))
    typeI_sum, _ = binomial.binomial_accumulator(lambda x: x[..., 0] >= thresh)(
        theta_tiles, is_null_per_arm, uniforms
    )

    typeI_est, typeI_CI = binomial.zero_order_bound(typeI_sum, nsims, delta, 1.0)
    typeI_bound = typeI_est + typeI_CI

    hob = binomial.holder_odi_bound(
        typeI_bound, theta_tiles, tile_corners, n_arm_samples, holderq
    )
    correct = {1: 0.04163348, 2: 0.37039003}
    np.testing.assert_allclose(hob[0], correct[d], atol=2e-6)

    pointwise_bound = binomial.invert_bound(
        hob, theta_tiles, tile_corners, n_arm_samples, holderq
    )
    np.testing.assert_allclose(pointwise_bound, typeI_bound, atol=1e-7)
