---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('imprint')
    language: python
    name: python3
---

```python
import sys
sys.path.append('../imprint/research/berry')
import berrylib.util as util
util.setup_nb()
```

```python
from berrylib.constants import Y_I2, N_I2
import berrylib.fast_inla as fast_inla
import berrylib.mcmc as mcmc
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

y=Y_I2
n=N_I2

fi = fast_inla.FastINLA(sigma2_n=50)
```

```python
def conditional_mu(mu, cov, i, x):
    """Return the conditional mean of a multivariate normal given a particular arm value.
    
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    """
    cov12 = np.delete(cov[i], i)
    cov22 = cov[i, i]
    mu1 = np.delete(mu, i)
    mu2 = mu[i]
    mu_cond = mu1 + cov12 / cov22 * (x - mu2)
    return mu_cond
```

```python
mcmc_results = []
for sig_idx in range(fi.sigma2_n):
    mcmc_results.append(mcmc.mcmc_berry(
        np.stack((y[-1:], n[-1:]), axis=-1),
        fi.logit_p1,
        fi.thresh_theta,
        dtype=np.float64,
        n_samples=200000,
        sigma2_val=fi.sigma2_rule.pts[sig_idx]
    ))
```

```python
mcmc_arm_marg = mcmc.mcmc_berry(
    np.stack((y[-1:], n[-1:]), axis=-1),
    fi.logit_p1,
    fi.thresh_theta,
    dtype=np.float64,
    n_samples=200000,
)
```

```python
def cond_laplace(t_i, sigma2_n):
    fi = fast_inla.FastINLA(sigma2_n=sigma2_n)
    data = np.stack((y, n), axis=-1)
    sigma2_post, _, theta_max, theta_sigma, hess_inv = fi.numpy_inference(data)
    arm_idx = 0
    theta_max_cond = np.empty((fi.sigma2_n, t_i.pts.shape[0], 4))
    theta_sigma_cond = np.empty((fi.sigma2_n, t_i.pts.shape[0], 4))
    p_t0_g_sig2_y = np.empty((fi.sigma2_n, t_i.pts.shape[0]))
    for sig_idx in range(fi.sigma2_n):
        for i, t0_val in enumerate(t_i.pts):
            cond_mu = conditional_mu(theta_max[-1, sig_idx], -hess_inv[-1, sig_idx], arm_idx, t0_val)

            theta_max_cond[sig_idx, i, 0] = t0_val
            theta_max_cond[sig_idx, i, 1:] = cond_mu

            exp_theta_adj = np.exp(theta_max_cond[sig_idx, i] + fi.logit_p1)
            C = 1.0 / (exp_theta_adj + 1)
            hess = fi.neg_precQ[sig_idx].copy()
            hess[np.arange(4), np.arange(4)] -= (n[-1] * exp_theta_adj * (C**2))

            cond_hess_inv = np.linalg.inv(hess)
            theta_sigma_cond[sig_idx, i] = np.sqrt(np.diagonal(-cond_hess_inv))

            logjoint = fi.model.log_joint(fi, data[-1:], theta_max_cond[None, sig_idx, i])[0,sig_idx]
            p_t0_g_sig2_y[sig_idx, i] = logjoint + 0.5 * np.log(np.linalg.det(-cond_hess_inv[1:, 1:]))
    laplace = np.exp(p_t0_g_sig2_y)
    laplace /= np.sum(laplace * t_i.wts[None,:], axis=1)[:, None]
    laplace_integral = np.sum(laplace * (sigma2_post[-1] * fi.sigma2_rule.wts)[:, None], axis=0)
    return theta_max, theta_sigma, laplace, laplace_integral
```

```python
t_i = util.simpson_rule(101, -15, 2)
theta_max, theta_sigma, laplace, laplace_integral = cond_laplace(t_i, 50)
```

```python
arm_idx = 0
plt.figure(figsize=(12, 4))
for i, sig_idx in enumerate([2, 25, 40, 48]):#range(fi.sigma2_n):
    mcmc_arm = mcmc_results[sig_idx]["x"][0]["theta"][0, :, arm_idx]
    mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, t_i.pts, t_i.wts)
    gaussian = scipy.stats.norm.pdf(t_i.pts, theta_max[-1, sig_idx, 0], theta_sigma[-1, sig_idx, 0])

    plt.subplot(1,5,1 + i)
    plt.title(f'$\sigma^2$ = {fi.sigma2_rule.pts[sig_idx]:5.2e}')
    plt.plot(t_i.pts, gaussian, 'k-')
    plt.plot(t_i.pts, laplace[sig_idx], 'r-')
    plt.plot(t_i.pts, mcmc_p_ti_g_y, 'b-')
plt.show()
```

```python
mcmc_arm = mcmc_arm_marg["x"][0]["theta"][0, :, arm_idx]
mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, t_i.pts, t_i.wts)
```

```python
from scipy.special import logit
```

```python
t_i = util.simpson_rule(101, -15, 2)
_, _, _, laplace_integral = cond_laplace(t_i, 20)
plt.plot(t_i.pts, laplace_integral, label=f'simpson101')
plt.plot(t_i.pts, mcmc_p_ti_g_y)

# t_i = util.gauss_rule(20, -15, 2)
# _, _, _, laplace_integral = cond_laplace(t_i, 20)
# plt.plot(t_i.pts, laplace_integral, label=f'gauss20')

# plt.vlines([logit(0.1) - logit(0.3), logit(0.2) - logit(0.3)], 0, 0.25, 'k')

plt.legend()
plt.show()
```

```python

```
