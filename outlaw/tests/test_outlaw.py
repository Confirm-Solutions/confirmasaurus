import time

import numpy as np
import pytest
import scipy.stats
from numpy import nan
from scipy.special import expit
from scipy.special import logit

import outlaw.berry as berry
import outlaw.inla as inla
import outlaw.numpyro_interface as numpyro_interface
import outlaw.quad as quad

# noreorder
import jax
import jax.numpy as jnp


def test_log_joint_from_numpyro():
    params = dict(sig2=10.0, theta=np.array([0, 0, 0]))
    data = np.array([[6.0, 35], [5, 35], [4, 35]])
    ll_fnc, _ = numpyro_interface.from_numpyro(berry.model(3), "sig2", (3, 2))
    ll = jax.jit(ll_fnc)(params, data)

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
    a, b = dict(sig2=10.0, theta=jnp.array([3, nan])), dict(
        sig2=None, theta=jnp.array([nan, 2])
    )
    out = inla.merge(a, b)
    assert out["sig2"] == 10.0
    np.testing.assert_allclose(out["theta"], [3, 2])


def test_ravel_fncs():
    tt = np.arange(4, dtype=np.float64)
    tt[1] = nan
    ex = dict(sig2=np.array([nan]), theta=tt)
    spec = inla.ParamSpec(ex)
    r = spec.ravel_f(ex)
    np.testing.assert_allclose(r, [0, 2, 3])
    ur = spec.unravel_f(r)
    assert ur["sig2"] is None
    np.testing.assert_allclose(ur["theta"], ex["theta"])


def test_pin_to_spec():
    full_params = dict(
        sig2=np.array([1.0]),
        theta=np.array([0.0, 0.0, 0.0, 0.0]),
    )

    for pin in ["sig2", ["sig2"], ("sig2", 0), [("sig2", 0)]]:
        spec = inla.ParamSpec(numpyro_interface.pin_params(full_params, pin))
        np.testing.assert_allclose(spec.param_example["sig2"], np.array([nan]))
        np.testing.assert_allclose(spec.param_example["theta"], np.array([0, 0, 0, 0]))

    spec = inla.ParamSpec(
        numpyro_interface.pin_params(full_params, [("sig2", 0), ("theta", 1)])
    )
    np.testing.assert_allclose(spec.param_example["sig2"], np.array([nan]))
    np.testing.assert_allclose(spec.param_example["theta"], np.array([0, nan, 0, 0]))


def test_grad_hess():
    data = np.array([[7, 35], [6.0, 35], [5, 35], [4, 35]])

    def grad_hess(p_ex, x, p_pinned, data):
        ops = inla.from_log_joint(berry.log_joint(4), p_ex)
        grad = ops.gradv(x[None], p_pinned, data)
        hess = ops.hessv(x[None], p_pinned, data)
        return grad, hess

    p_ex = dict(
        sig2=np.array([10.0]),
        theta=np.array([0.0, 0.0, 0, 0]),
    )
    grad, hess = grad_hess(
        p_ex, np.array([10.0, 0, 0, 0, 0]), dict(sig2=None, theta=None), data
    )
    full_grad = np.array([-0.25124812, -3.5032682, -4.5032682, -5.5032682, -6.5032682])
    hess01 = np.array(
        [
            [2.5007863e-02, 7.9709353e-06, 7.9717356e-06, 7.9713354e-06, 7.9718011e-06],
            [7.9710153e-06, -7.4256096e00, 2.4390254e-02, 2.4390249e-02, 2.4390247e-02],
        ]
    )
    np.testing.assert_allclose(grad[0], full_grad, rtol=1e-4)
    np.testing.assert_allclose(hess[0, :2], hess01, rtol=1e-4)

    grad, hess = grad_hess(
        dict(sig2=jnp.array([nan]), theta=jnp.array([0, nan, 0, 0])),
        np.array([0, 0, 0, 0]),
        dict(sig2=np.array([10.0]), theta=jnp.array([[nan, 0.0, nan, nan]])),
        data,
    )
    np.testing.assert_allclose(grad[0], full_grad[[1, 3, 4]], rtol=1e-4)
    np.testing.assert_allclose(hess[0, 0], hess01[1, [1, 3, 4]], rtol=1e-4)


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
    data = berry.figure2_data().astype(dtype)
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts.astype(dtype)
    inla_ops = berry.optimized(sig2, dtype=dtype).config(
        max_iter=10, opt_tol=dtype(1e-6)
    )
    logpost, x_max, _, iters = jax.jit(
        jax.vmap(inla_ops.laplace_logpost, in_axes=(None, None, 0))
    )(np.zeros((15, 4), dtype=dtype), dict(sig2=sig2), data)

    post = inla.exp_and_normalize(logpost, sig2_rule.wts.astype(dtype)[None, :], axis=1)

    np.testing.assert_allclose(x_max[0, 12], xmax0_12, rtol=1e-3)
    np.testing.assert_allclose(
        post[0], sig2_post, rtol=5e-3 if dtype is np.float64 else 5e-2
    )
    assert post.dtype == dtype
    assert x_max.dtype == dtype


@pytest.mark.parametrize("n_arms", [2, 3, 4])
def test_gaussian_laplace(n_arms):
    dtype = np.float64
    data = berry.figure2_data(1).astype(dtype)[:, :n_arms]
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts.astype(dtype)
    p_pinned = dict(sig2=sig2, theta=None)

    def logpost(ops):
        f = jax.jit(jax.vmap(custom_ops.laplace_logpost, in_axes=(None, None, 0)))
        logpost, x_max, _, iters = f(
            np.zeros((15, n_arms), dtype=dtype), p_pinned, data
        )
        return (
            inla.exp_and_normalize(
                logpost, sig2_rule.wts.astype(dtype)[None, :], axis=1
            ),
            x_max,
        )

    custom_ops = berry.optimized(sig2, n_arms=n_arms, dtype=dtype).config(
        opt_tol=dtype(1e-3)
    )
    custom_post, custom_x_max = logpost(custom_ops)

    # Compare the custom outputs against the version that uses automatic differentiation
    p_ex = dict(sig2=np.array([nan]), theta=np.zeros(n_arms))
    ad_ops = inla.from_log_joint(berry.log_joint(n_arms), p_ex)
    ad_post, ad_x_max = logpost(ad_ops)
    np.testing.assert_allclose(
        custom_x_max,
        ad_x_max,
        atol=1e-3,
    )
    np.testing.assert_allclose(custom_post, ad_post, rtol=5e-3)


def compare_arm0_against_mcmc(cur_loc, arm_marg, cx, wts, exact):
    """Compare the arm 0 posterior against the stored MCMC results using
    Jensen-Shannon divergence.
    """
    mcmc_test_data = np.load(cur_loc.joinpath("test_conditional_inla.npy"))
    divergence = inla.jensen_shannon_div(
        arm_marg[:, 0, :], mcmc_test_data[..., 1], wts[:, 0, :], axis=0
    )
    # These are divergence values produced by a run of the conditional laplace
    # approximation.
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
    if exact:
        np.testing.assert_allclose(divergence, target_divergence, rtol=1e-3)
    else:
        assert np.all(divergence < 2 * target_divergence)


def test_full_laplace(cur_loc):
    n_arms = 4
    data = berry.figure2_data(1)[:, :n_arms]
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts
    p_ex = dict(sig2=np.array([nan]), theta=np.zeros(n_arms))
    ad_ops = inla.from_log_joint(berry.log_joint(n_arms), p_ex)
    p_pinned = dict(sig2=sig2, theta=None)
    _, x_max, hess, _ = jax.jit(
        jax.vmap(ad_ops.laplace_logpost, in_axes=(None, None, 0))
    )(np.zeros((15, n_arms)), p_pinned, data)

    arm_idx = 0
    theta_arm_ex = np.zeros(4)
    theta_arm_ex[arm_idx] = nan
    arm_ex = dict(sig2=np.array([nan]), theta=theta_arm_ex)
    ad_arm_ops = inla.from_log_joint(berry.log_joint(n_arms), arm_ex)

    inv_hess = jnp.linalg.inv(hess)
    cx, wts = inla.gauss_hermite_grid(x_max, inv_hess[:, :, arm_idx], arm_idx, n=25)

    sig2_tiled = np.tile(sig2[None, None, :], (cx.shape[0], data.shape[0], 1))
    theta_fixed = np.full((*sig2_tiled.shape, 4), nan)
    theta_fixed[..., arm_idx] = cx
    arm_pinned = dict(sig2=sig2_tiled, theta=theta_fixed)

    x0 = jnp.delete(x_max, arm_idx, -1)
    laplace_f = jax.jit(
        jax.vmap(
            jax.vmap(ad_arm_ops.laplace_logpost, in_axes=(0, 0, 0)),
            in_axes=(None, 0, None),
        )
    )
    logpost_arm, arm_x_max, arm_hess, arm_iters = laplace_f(x0, arm_pinned, data)
    arm_marg = inla.exp_and_normalize(logpost_arm, wts, axis=0)

    compare_arm0_against_mcmc(cur_loc, arm_marg, cx, wts, exact=False)


def test_conditional_inla(cur_loc):
    # The origin of this test is in conditional_inla.ipynb Take a look there
    # for figures showing density comparisons.

    # Step 1) Set up the data and problem, hyperparam posterior
    n_arms = 4
    data = berry.figure2_data(1)[:, :n_arms]
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts
    p_ex = dict(sig2=np.array([nan]), theta=np.zeros(n_arms))
    ad_ops = inla.from_log_joint(berry.log_joint(n_arms), p_ex)
    p_pinned = dict(sig2=sig2, theta=None)
    _, x_max, hess, _ = jax.jit(
        jax.vmap(ad_ops.laplace_logpost, in_axes=(None, None, 0))
    )(np.zeros((15, n_arms)), p_pinned, data)

    arm_idx = 0
    theta_arm_ex = np.zeros(4)
    theta_arm_ex[arm_idx] = nan
    arm_ex = dict(sig2=np.array([nan]), theta=theta_arm_ex)
    ad_arm_ops = inla.from_log_joint(berry.log_joint(n_arms), arm_ex)

    inv_hess = jnp.linalg.inv(hess)
    cx, wts = inla.gauss_hermite_grid(x_max, inv_hess[:, :, arm_idx], arm_idx, n=25)

    sig2_tiled = np.tile(sig2[None, None, :], (cx.shape[0], data.shape[0], 1))
    theta_fixed = np.full((*sig2_tiled.shape, 4), nan)
    theta_fixed[..., arm_idx] = cx
    arm_pinned = dict(sig2=sig2_tiled, theta=theta_fixed)

    cond_laplace_f = jax.jit(
        jax.vmap(
            jax.vmap(ad_arm_ops.cond_laplace_logpost, in_axes=(0, 0, 0, 0, 0, None)),
            in_axes=(None, None, 0, None, 0, None),
        ),
        static_argnums=(5,),
    )
    logpost_arm = cond_laplace_f(
        x_max, inv_hess[:, :, arm_idx], arm_pinned, data, cx, arm_idx
    )
    arm_marg = inla.exp_and_normalize(logpost_arm, wts, axis=0)
    compare_arm0_against_mcmc(cur_loc, arm_marg, cx, wts, exact=True)


def test_solve_inv_basket():
    np.random.seed(10)
    b = np.random.rand(1)[0]
    for i in range(3):
        for d in range(2, 10):
            a = np.random.rand(d)
            m = np.full((d, d), b) + np.diag(a)

            # Test inverse:
            minv = np.linalg.inv(m)
            np.testing.assert_allclose(berry.inv_basket(a, b), minv)

            # Test solve:
            v = np.random.rand(d)
            correct = np.linalg.solve(m, v)
            x, denom = berry.solve_basket(a, b, v)
            np.testing.assert_allclose(x, correct)

            # Test logdet:
            logdet = berry.logdet_basket((a, denom))
            np.testing.assert_allclose(logdet, np.linalg.slogdet(m)[1])


def my_timeit(N, f, iter=5, inner_iter=10, should_print=True):
    _ = f()
    runtimes = []
    for i in range(iter):
        start = time.time()
        f()
        runtimes.append(time.time() - start)
    if should_print:
        print("median runtime", np.median(runtimes))
        print("min us per sample ", np.min(runtimes) * 1e6 / N)
        print("median us per sample", np.median(runtimes) * 1e6 / N)
    return runtimes


def benchmark(N=10000, iter=5):
    dtype = np.float32
    data = berry.figure2_data(N).astype(dtype)
    sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
    sig2 = sig2_rule.pts.astype(dtype)
    x0 = jnp.zeros((sig2.shape[0], 4), dtype=dtype)

    print("\ncustom dirty bayes")
    db = jax.jit(jax.vmap(berry.build_dirty_bayes(sig2, n_arms=4, dtype=dtype)))
    my_timeit(N, lambda: db(data)[0].block_until_ready(), iter=iter)

    def bench_ops(name, ops):
        print(f"\n{name} gaussian")
        hyperpost = jax.jit(jax.vmap(ops.laplace_logpost, in_axes=(None, None, 0)))
        p_pinned = dict(sig2=sig2, theta=None)
        my_timeit(
            N, lambda: hyperpost(x0, p_pinned, data)[0].block_until_ready(), iter=iter
        )

        print(f"\n{name} laplace")
        _, x_max, hess_info, _ = hyperpost(x0, p_pinned, data)
        arm_logpost_f = jax.jit(
            jax.vmap(
                jax.vmap(
                    ops.cond_laplace_logpost, in_axes=(0, 0, None, 0, 0, None, None)
                ),
                in_axes=(None, None, None, None, 0, None, None),
            ),
            static_argnums=(5, 6),
        )
        invv = jax.jit(jax.vmap(jax.vmap(ops.invert)))

        def f():
            inv_hess = invv(hess_info)
            arm_post = []
            for arm_idx in range(4):
                cx, wts = inla.gauss_hermite_grid(
                    x_max, inv_hess[..., arm_idx, :], arm_idx, n=25
                )
                arm_logpost = arm_logpost_f(
                    x_max, inv_hess[:, :, arm_idx], p_pinned, data, cx, arm_idx, True
                )
                arm_post.append(inla.exp_and_normalize(arm_logpost, wts, axis=0))
            return jnp.array(arm_post)

        f_jit = jax.jit(f)
        my_timeit(N, lambda: f_jit().block_until_ready(), iter=iter)

    custom_ops = berry.optimized(sig2, dtype=dtype).config(max_iter=10)
    bench_ops("custom berry", custom_ops)

    ad_ops = inla.from_log_joint(
        berry.log_joint(4),
        dict(sig2=np.array([nan], dtype=dtype), theta=np.full(4, 0.0, dtype=dtype)),
    ).config(max_iter=10)
    bench_ops("numpyro berry", ad_ops)


if __name__ == "__main__":
    # set to cpu or gpu to run on a specific device.
    jax.config.update("jax_platform_name", "gpu")

    benchmark(N=10000, iter=1)
    # Running with N=1 is useful for benchmarking the JIT.
    # benchmark(N=1, iter=1)
