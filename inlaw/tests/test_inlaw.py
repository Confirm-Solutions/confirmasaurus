import timeit

if "profile" not in __builtins__:
    __builtins__.profile = lambda x: x

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from jax.config import config
from scipy.special import expit
from scipy.special import logit

import inlaw.berry_model as berry_model
import inlaw.inla as inla
import inlaw.quad as quad
import inlaw.smalljax as smalljax

config.update("jax_enable_x64", True)


def test_log_likelihood():
    params = dict(sig2=10.0, theta=np.array([0, 0, 0]))
    data = np.array([[6.0, 35], [5, 35], [4, 35]])
    ll = inla.build_log_likelihood(berry_model.berry_model(3))(
        params, dict(sig2=None, theta=None), data
    )

    mu_0 = -1.34
    mu_sig2 = 100
    sig2_alpha = 0.0005
    sig2_beta = 0.000005
    invgamma_term = scipy.stats.invgamma.logpdf(
        params["sig2"], sig2_alpha, scale=sig2_beta
    )
    cov = jnp.full((3, 3), mu_sig2) + jnp.diag(jnp.repeat(params["sig2"], 3))
    normal_term = scipy.stats.multivariate_normal.logpdf(
        params["theta"], np.repeat(mu_0, 3), cov
    )
    binomial_term = scipy.stats.binom.logpmf(
        data[..., 0], data[..., 1], expit(params["theta"] + logit(0.3))
    )
    np.testing.assert_allclose(
        ll, invgamma_term + normal_term + sum(binomial_term), rtol=1e-6
    )


def test_merge():
    a, b = dict(sig2=10.0, theta=jnp.array([3, np.nan])), dict(
        sig2=None, theta=jnp.array([np.nan, 2])
    )
    out = inla.merge(a, b)
    assert out["sig2"] == 10.0
    np.testing.assert_allclose(out["theta"], [3, 2])


def test_ravel_fncs():
    tt = np.arange(4, dtype=np.float64)
    tt[1] = np.nan
    ex = dict(sig2=np.array([np.nan]), theta=tt)
    spec = inla.ParamSpec(ex)
    r = spec.ravel_f(ex)
    np.testing.assert_allclose(r, [0, 2, 3])
    ur = spec.unravel_f(r)
    assert ur["sig2"] is None
    np.testing.assert_allclose(ur["theta"], ex["theta"])


def test_pin_to_spec():
    model = berry_model.berry_model(4)
    data_ex = np.ones((4, 2))

    for pin in ["sig2", ["sig2"], ("sig2", 0), [("sig2", 0)]]:
        spec = inla.pin_to_spec(model, pin, data_ex)
        np.testing.assert_allclose(spec.param_example["sig2"], np.array([np.nan]))
        np.testing.assert_allclose(spec.param_example["theta"], np.array([0, 0, 0, 0]))

    spec = inla.pin_to_spec(model, [("sig2", 0), ("theta", 1)], data_ex)
    np.testing.assert_allclose(spec.param_example["sig2"], np.array([np.nan]))
    np.testing.assert_allclose(spec.param_example["theta"], np.array([0, np.nan, 0, 0]))


def test_grad_hess():
    data = np.array([[7, 35], [6.0, 35], [5, 35], [4, 35]])
    spec = inla.ParamSpec(
        dict(
            sig2=np.array([10.0]),
            theta=np.array([0.0, 0.0, 0, 0]),
        )
    )
    ll_fnc = inla.build_log_likelihood(berry_model.berry_model(4))
    grad_hess_vmap = inla.build_grad_hess(ll_fnc, spec)
    grad, hess = grad_hess_vmap(
        np.concatenate((np.full((1, 1, 1), 10.0), np.zeros((1, 1, 4))), axis=2),
        dict(sig2=None, theta=None),
        data[None],
    )
    full_grad = np.array([-0.25124812, -3.5032682, -4.5032682, -5.5032682, -6.5032682])
    hess01 = np.array(
        [
            [2.5007863e-02, 7.9709353e-06, 7.9717356e-06, 7.9713354e-06, 7.9718011e-06],
            [7.9710153e-06, -7.4256096e00, 2.4390254e-02, 2.4390249e-02, 2.4390247e-02],
        ]
    )
    np.testing.assert_allclose(grad[0, 0], full_grad, rtol=1e-4)
    np.testing.assert_allclose(hess[0, 0, :2], hess01, rtol=1e-4)

    spec2 = inla.ParamSpec(
        dict(sig2=jnp.array([jnp.nan]), theta=jnp.array([0, np.nan, 0, 0]))
    )
    grad_hess_vmap = inla.build_grad_hess(ll_fnc, spec2)
    theta_fixed = jnp.array([np.nan, 0.0, np.nan, np.nan])
    grad, hess = grad_hess_vmap(
        np.zeros((2, 1, 4)),
        dict(sig2=np.array([[10.0]]), theta=theta_fixed[None]),
        np.tile(data[None], (2, 1, 1)),
    )
    np.testing.assert_allclose(grad[0, 0], full_grad[[1, 3, 4]], rtol=1e-4)
    np.testing.assert_allclose(hess[0, 0, 0], hess01[1, [1, 3, 4]], rtol=1e-4)


xmax0_12 = np.array([-6.04682818, -2.09586893, -0.21474981, -0.07019088])
sig2_post = np.array(
    [
        1.25954474e02,
        4.52520893e02,
        8.66625278e02,
        5.08333300e02,
        1.30365045e02,
        2.20403048e01,
        3.15183578e00,
        5.50967224e-01,
        2.68365061e-01,
        1.23585852e-01,
        1.13330444e-02,
        5.94800210e-04,
        4.01075571e-05,
        4.92782335e-06,
        1.41605356e-06,
    ]
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_berry(dtype):
    data = berry_model.figure2_data().astype(dtype)
    # fl = inla.FullLaplace(berry_model.berry_model(4), "sig2", data[0])
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts.astype(dtype)
    fl = berry_model.fast_berry(sig2, dtype=dtype, max_iter=10, tol=dtype(1e-6))
    logpost, x_max, _, iters = fl(dict(sig2=sig2), data, jit=False, should_batch=False)
    post = inla.exp_and_normalize(logpost, sig2_rule.wts.astype(dtype)[None, :], axis=1)

    np.testing.assert_allclose(x_max[0, 12], xmax0_12, rtol=1e-3)
    np.testing.assert_allclose(
        post[0], sig2_post, rtol=5e-3 if dtype is np.float64 else 5e-2
    )
    assert post.dtype == dtype
    assert x_max.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_arms", [2, 3, 4])
def test_full_laplace(dtype, n_arms):
    data = berry_model.figure2_data(1).astype(dtype)[:, :n_arms]
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts.astype(dtype)
    laplace = berry_model.fast_berry(sig2, n_arms=n_arms, tol=1e-6)
    logpost_custom, x_max_custom, hess_custom, _ = laplace(
        dict(sig2=sig2), data, jit=False, should_batch=False
    )
    post_custom = inla.exp_and_normalize(
        logpost_custom, sig2_rule.wts.astype(dtype)[None, :], axis=1
    )

    # Compare the custom outputs against the
    fl = inla.FullLaplace(berry_model.berry_model(n_arms), "sig2", data[0], tol=1e-6)
    logpost, x_max, hess, _ = fl(
        dict(sig2=sig2.astype(np.float64)),
        data.astype(np.float64),
        jit=False,
    )
    post = inla.exp_and_normalize(logpost, sig2_rule.wts.astype(dtype)[None, :], axis=1)
    np.testing.assert_allclose(
        x_max_custom,
        x_max,
        atol=1e-3,
    )
    np.testing.assert_allclose(post_custom, post, rtol=5e-3)


def test_conditional_inla(cur_loc):
    # The origin of this test is in conditional_inla.ipynb Take a look there
    # for figures showing density comparisons.
    n_arms = 4
    data = berry_model.figure2_data(1)[:, :n_arms]
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    fl = inla.FullLaplace(berry_model.berry_model(n_arms), "sig2", data[0])
    p_pinned = dict(sig2=sig2_rule.pts, theta=None)
    logpost, x_max, hess, _ = fl(p_pinned, data)
    # post = inla.exp_and_normalize(logpost, sig2_rule.wts[None, :], axis=1)

    arm_idx = 0
    cond_inla_f = inla.build_conditional_inla(fl.log_joint_single, fl.spec)
    cx, wts = inla.gauss_hermite_grid(x_max, hess, arm_idx, n=25)
    lp = cond_inla_f(x_max, p_pinned, data, hess, cx, arm_idx)
    arm_marg = inla.exp_and_normalize(lp, wts, axis=0)

    mcmc_test_data = np.load(cur_loc.joinpath("test_conditional_inla.npy"))

    divergence = inla.jensen_shannon_div(
        arm_marg[:, 0, :], mcmc_test_data[..., 1], wts[:, 0, :], axis=0
    )
    print(divergence)
    target_divergence = np.array(
        [
            1.04143232e-04,
            3.07438844e-04,
            6.45982416e-05,
            9.89656738e-05,
            7.83444433e-05,
            4.13895995e-05,
            2.81239500e-05,
            1.96616409e-05,
            3.00670662e-05,
            2.55080402e-05,
            7.19278681e-05,
            1.96892343e-04,
            1.42703595e-03,
            1.01020761e-04,
            1.32290054e-04,
        ]
    )
    np.testing.assert_allclose(cx[:, 0, :], mcmc_test_data[..., 0])
    np.testing.assert_allclose(divergence, target_divergence, rtol=1e-3)


def test_solve_basket():
    np.random.seed(10)
    b = np.random.rand(1)[0]
    for i in range(3):
        for d in range(2, 10):
            a = np.random.rand(d)
            m = np.full((d, d), b) + np.diag(a)
            v = np.random.rand(d)
            correct = np.linalg.solve(m, v)
            x, denom = berry_model.solve_basket(a, b, v)
            print(x, correct)
            np.testing.assert_allclose(x, correct)
            logdet = berry_model.logdet((a, denom))
            np.testing.assert_allclose(logdet, np.linalg.slogdet(m)[1])


def my_timeit(N, f, should_print=True):
    _ = f()
    number = 5
    runtimes = np.array(timeit.repeat(f, number=number)) / number
    if should_print:
        print("median runtime", np.median(runtimes))
        print("min us per sample ", np.min(runtimes) * 1e6 / N)
        print("median us per sample", np.median(runtimes) * 1e6 / N)
    return runtimes


def benchmark_custom():
    N = 1000
    dtype = np.float32
    data = berry_model.figure2_data(N).astype(dtype)
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts.astype(dtype)
    laplace = berry_model.fast_berry(sig2, dtype=dtype, max_iter=10, tol=1e-2)
    print("custom")
    my_timeit(N, lambda: laplace(dict(sig2=sig2), data, should_batch=False)[0])

    print("\ngeneric")
    fl = inla.FullLaplace(berry_model.berry_model(4), "sig2", data[0])
    my_timeit(
        N,
        lambda: fl(
            dict(sig2=sig2.astype(np.float64)),
            data.astype(np.float64),
            should_batch=False,
        )[0],
    )


@profile
def benchmark_jit():
    n_arms = 4
    data = berry_model.figure2_data(1)[:, :n_arms]
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)

    fl = inla.FullLaplace(berry_model.berry_model(n_arms), "sig2", data[0])
    p_pinned = dict(sig2=sig2_rule.pts, theta=None)

    x0 = jnp.zeros((data.shape[0], 15, 4), dtype=data.dtype)
    x_max, hess_info, iters = jax.jit(fl.optimizer)(x0, p_pinned, data)

    logdet = jax.vmap(jax.vmap(lambda x: smalljax.logdet(-x)))

    def calc_log_posterior(x_max, hess_info, p_pinned, data):
        lj = fl.log_joint(x_max, p_pinned, data)
        log_post = lj - x_max.dtype.type(0.5) * logdet(hess_info)
        return log_post

    jax.jit(calc_log_posterior)(x_max, hess_info, p_pinned, data)


if __name__ == "__main__":
    # benchmark_custom()
    benchmark_jit()
