```python
# this sets up autoreload and some plotting configurations
import outlaw.nb_util as nb_util

nb_util.setup_nb()
```

```python
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
```

```python
# to improve this, we should grid in the natural parameters:
# grid in sig1/sig2
# grid in (mu1 - mu2) / sqrt(sig1^2 + sig2^2)
```

```python
# simulate 20 gaussians as our data
n = 20


def simulate(nsims, mu, sig):
    arm1 = np.random.normal(0, 1, size=(nsims, n))
    arm2 = np.random.normal(mu, sig, size=(nsims, n))
    return arm1, arm2


# we're going to use the scipy.stats.ttest_ind function as the baseline correct
# answer to make sure the jax implementation is correct
def compute_rejections(arm1, arm2):
    res = scipy.stats.ttest_ind(arm1, arm2, equal_var=False, axis=1)
    reject = res[1] < 0.05
    n_rejections = np.sum(reject)
    return n_rejections
```

```python
def simulate_jax(key, nsims, mu, sig):
    key1, key2 = jax.random.split(key, 2)
    arm1 = jax.random.normal(key1, shape=(nsims, n))
    arm2 = jax.random.normal(key2, shape=(nsims, n)) * sig + mu
    return arm1, arm2


def jax_ttest(arm1, arm2):
    """
    Implementing the two tailed t-test fully in JAX. The scipy.stats.ttest_ind
    function provided a template for this code. Look there for code on
    implementing a one-sided test and many other variants.
    """
    n1 = arm1.shape[0]
    n2 = arm2.shape[0]
    mu1 = jnp.mean(arm1)
    mu2 = jnp.mean(arm2)
    v1 = jnp.var(arm1, ddof=1)
    v2 = jnp.var(arm2, ddof=1)
    vn1 = v1 / n1
    vn2 = v2 / n2

    df = (vn1 + vn2) ** 2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))
    df = jnp.where(jnp.isnan(df), 1, df)
    denom = jnp.sqrt(vn1 + vn2)

    # two-tailed test
    t = (mu1 - mu2) / denom
    # p = scipy.special.stdtr(df, -np.abs(t)) * 2
    x = df / (t**2 + df)
    p = jax.scipy.special.betainc(df / 2, 0.5, x)
    return t, p


arm1, arm2 = simulate(1, 0.1, 0.9)
scipy_res = scipy.stats.ttest_ind(arm1, arm2, equal_var=False, axis=1)
my_t, my_p = jax.jit(jax_ttest)(arm1[0], arm2[0])
np.testing.assert_allclose(my_t, scipy_res[0], atol=1e-5)
np.testing.assert_allclose(my_p, scipy_res[1], atol=1e-5)
```

```python
def sim_and_compute(key, nsims, mu2, sig2):
    """Using the simulate and ttest function to compute the number of rejections"""
    arms = simulate_jax(key, nsims, mu2, sig2)
    _, p = jax.vmap(jax_ttest)(*arms)
    reject = p < 0.05
    n_rejections = jnp.sum(reject)
    return n_rejections


nsims = 10000
# Construct a grid mu in [-1, 2] and sig in [0.1, 3]
mus = np.linspace(-1, 2, 31)
sigs = np.linspace(0.1, 3, 30)
jax_reject_f = jax.jit(
    jax.vmap(
        jax.vmap(sim_and_compute, in_axes=(0, None, None, 0)),
        in_axes=(0, None, 0, None),
    ),
    static_argnums=(1,),
)
keys = jax.random.split(jax.random.PRNGKey(0), mus.shape[0] * sigs.shape[0]).reshape(
    (mus.shape[0], sigs.shape[0], 2)
)
reject_grid = jax_reject_f(keys, nsims, mus, sigs)
```

```python
%%timeit
jax_reject_f(keys, nsims, mus, sigs)
```

```python
# The grid is in terms of number of rejections out of the maximum nsims
plt.imshow(reject_grid)
plt.colorbar()
plt.show()
```

```python
# Now let's test our interpolation! We'll select 2000 random points and create
# a distribution of the error for these points.
n_test = 2000
rand_pts = np.random.uniform(size=(n_test, 2))
# The sigma
rand_pts[:, 1] += 0.1
```

```python
# We'll compute two simulation-based values for each test point:
# 1) The test_reject value. We compute with 50x more simulations than we used for constructing the table.
# 2) The baseline_reject value. We compute with the same number of simulations as we used for constructing the table.
nsims_test = nsims * 50
test_reject = np.empty(n_test)
keys_test = jax.random.split(jax.random.PRNGKey(1), n_test)
jax_test_f = jax.jit(
    jax.vmap(sim_and_compute, in_axes=(0, None, 0, 0)), static_argnums=(1,)
)
test_reject = np.empty(n_test)
baseline_reject = np.empty(n_test)
chunk_size = 10
for i in range(0, n_test, chunk_size):
    end = min(n_test, i + chunk_size)
    test_reject[i:end] = jax_test_f(
        keys_test[i:end], nsims_test, rand_pts[i:end, 0], rand_pts[i:end, 1]
    )
    baseline_reject[i:end] = jax_test_f(
        keys_test[i:end], nsims, rand_pts[i:end, 0], rand_pts[i:end, 1]
    )
```

```python
# Now do three types of interpolation:
# 1) Linear interpolation
# 2) Spline interpolation
# 3) RBF interpolation

interp_reject_linear = scipy.interpolate.interpn(
    (mus, sigs), reject_grid, rand_pts, method="linear"
)
interp_reject_spline = scipy.interpolate.interpn(
    (mus, sigs), reject_grid, rand_pts, method="splinef2d"
)
mu_grid, sig_grid = np.meshgrid(mus, sigs, indexing="ij")
grid = np.stack((mu_grid.ravel(), sig_grid.ravel()), axis=1)

interp_rbf = scipy.interpolate.RBFInterpolator(
    grid,
    reject_grid.ravel(),
    neighbors=30,  # , kernel="quintic"
)(rand_pts)
```

```python
# Finally, plot a comparison of the error density functions resulting from the
# three interpolations and from the baseline rejection method. The baseline
# rejection method is a useful measure of how much error there would be if we
# did not use any interpolation.
from scipy.stats import gaussian_kde

linear_err = (test_reject / nsims_test) - (interp_reject_linear / nsims)
spline_err = (test_reject / nsims_test) - (interp_reject_spline / nsims)
rbf_err = (test_reject / nsims_test) - (interp_rbf / nsims)
baseline_err = (test_reject / nsims_test) - (baseline_reject / nsims)
xs = np.linspace(-0.015, 0.015, 200)
plt.plot(xs, gaussian_kde(linear_err)(xs), label="linear")
plt.plot(xs, gaussian_kde(spline_err)(xs), label="spline")
plt.plot(xs, gaussian_kde(rbf_err)(xs), label="rbf")
plt.plot(xs, gaussian_kde(baseline_err)(xs), "k-", label="sims")
plt.legend()
plt.ylabel("density")
plt.xlabel("fractional error in rejection rate")
plt.show()
```

## Junk

```python
# %%time
# nsims = 10000
# mus = np.linspace(-1, 2, 31)
# sigs = np.linspace(0.1, 3, 30)
# reject_grid = np.empty((mus.shape[0], sigs.shape[0]))
# for i, m in enumerate(mus):
#     for j, s in enumerate(sigs):
#         reject_grid[i, j] = compute_rejections(*simulate(nsims, m, s))
```

```python
# sample_mean1 = np.mean(arm1, axis=1)
# sample_mean2 = np.mean(arm2, axis=1)
# std_err1 = np.std(arm1, axis=1) / np.sqrt(n)
# std_err2 = np.std(arm2, axis=1) / np.sqrt(n)
# T = (sample_mean1 - sample_mean2) / np.sqrt(std_err1**2 + std_err2**2)
```

```python
# kde_reject = scipy.stats.gaussian_kde(dataset.T, weights=reject_grid.ravel())
# kde_interp_reject = kde_reject(rand_pts.T)
# kde_err = (sample_reject / nsims_test) - (kde_interp_reject / nsims)
# plt.plot(xs, gaussian_kde(kde_err)(xs), label='kde')
```
