---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.10.5 ('confirm')
    language: python
    name: python3
---

```python
import confirm.outlaw.nb_util as util
util.setup_nb()
```

```python
import confirm.outlaw.berry as berry
import confirm.outlaw.quad as quad
import confirm.outlaw.inla as inla
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
```

## Run Outlaw hyperparameter posterior

```python
dtype = np.float64
# data = berry.figure1_data(N=1)[0].astype(dtype)
data = berry.figure2_data(N=1)[0].astype(dtype)
sig2_rule = quad.log_gauss_rule(70, 1e-6, 1e3)
# sig2_rule = quad.log_gauss_rule(70, 1e-3, 1e3)
sig2 = sig2_rule.pts
# sig2_df = pd.DataFrame(dict(theta=sig2_rule.pts, wts=sig2_rule.wts))
# sig2_df.to_csv('sig2_rule.csv', index=False)
```

```python
# data = berry.figure1_data(N=1)[0].astype(dtype)
# data
```

```python
inla_ops = berry.optimized(sig2, dtype=dtype).config(
    max_iter=20, opt_tol=dtype(1e-9)
)

logpost, x_max, hess_info, iters = jax.jit(
    inla_ops.laplace_logpost
)(np.zeros((sig2.shape[0], 4), dtype=dtype), dict(sig2=sig2), data)
post = inla.exp_and_normalize(logpost, sig2_rule.wts.astype(dtype), axis=0)
```

```python
arm_logpost_f = jax.jit(
    jax.vmap(
        inla_ops.cond_laplace_logpost,
        in_axes=(None, None, None, None, 0, None, None),
    ),
    static_argnums=(5, 6),
)
invv = jax.jit(jax.vmap(inla_ops.invert))

inv_hess = invv(hess_info)
arm_idx = 0
cx, wts = inla.gauss_hermite_grid(
    x_max, inv_hess[..., arm_idx, :], arm_idx, n=25
)
# cx, wts = inla.latent_grid(
#     x_max, inv_hess[..., arm_idx, :], arm_idx, 
#     quad.gauss_rule(105, a=-5, b=5)
# )
cx = cx[:,0]
wts = wts[:,0]

arm_logpost = arm_logpost_f(
    x_max, inv_hess[:, :, arm_idx], dict(sig2=sig2, theta=None), data, cx, arm_idx, True
)
arm_post = inla.exp_and_normalize(arm_logpost, wts, axis=0)
arm_quad = quad.simpson_rule(101, -15, 2)
# arm_quad = quad.simpson_rule(101, -4, 2)
```

```python
interp_vmap = jax.vmap(lambda *args: jnp.interp(*args, 0, 0), in_axes=(None, 1, 1))
interp_marg_condsig2 = interp_vmap(arm_quad.pts, cx, arm_post).T
uncon_pdf_interp = np.sum(interp_marg_condsig2 * post * sig2_rule.wts, axis=1)
plt.plot(arm_quad.pts, uncon_pdf_interp, 'k-.')
plt.show()
```

```python
arm_idx = 0
inv_hess = jax.vmap(inla_ops.invert)(hess_info)

cx = np.tile(arm_quad.pts[:, None], (1, sig2.shape[0]))
wts = np.tile(arm_quad.wts[:, None], (1, sig2.shape[0]))
cond_laplace_f = jax.vmap(
    lambda *args: inla_ops.cond_laplace_logpost(*args, reduced=True),
    in_axes=(None, None, None, None, 0, None),
)
logpost_arm = cond_laplace_f(
    x_max, inv_hess[:, :, arm_idx], dict(sig2=sig2), data, cx, arm_idx
)
arm_post = inla.exp_and_normalize(logpost_arm, wts, axis=0)

group_size=20
plt.figure(figsize=(10, 10))
for subplot_idx, i0 in enumerate(range(0, sig2_rule.pts.shape[0], group_size)):
    plt.subplot(2,2,subplot_idx + 1)
    for i in range(i0, min(sig2_rule.pts.shape[0], i0 + group_size), 3):
        plt.plot(cx[:, i], arm_post[:, i], label='sig2={:.2e}'.format(sig2_rule.pts[i]))
    plt.legend(fontsize=8)
    # plt.xlim([-10, 1])
plt.show()
```

```python
theta_sigma = np.sqrt(np.diagonal(-inv_hess, axis1=1, axis2=2))
theta_mu = x_max
uncon_pdf_gaussian = np.sum(scipy.stats.norm.pdf(
    arm_quad.pts[:, None],
    theta_mu[None, :, arm_idx],
    theta_sigma[None, :, arm_idx],
) * post * sig2_rule.wts, axis=1)
```

## Dirty Bayes

```python
import confirm.berrylib.dirty_bayes as dirty_bayes

uncon_cdf_db = np.empty_like(uncon_pdf_interp)
for i, thresh in enumerate(arm_quad.pts):
    db_out = dirty_bayes.calc_dirty_bayes(
        np.where(data[None, ..., 0] == 0, 0.1, data[None, ..., 0]),
        data[None, ..., 1],
        berry.mu_0,
        berry.logit_p1,
        np.full((1, 4), thresh),
        sig2_rule,
    )
    uncon_cdf_db[i] = db_out['exceedance'][0,0]
pdf = (uncon_cdf_db[2:] - uncon_cdf_db[:-2]) / (arm_quad.pts[2:] - arm_quad.pts[:-2])
uncon_pdf_db = -np.concatenate(([0], pdf, [0])) 
uncon_pdf_db
```

## Compare latent marginals

```python
import confirm.berrylib.mcmc as mcmc
mcmc_results = mcmc.mcmc_berry(data[None], n_samples=int(1000000))
```

```python
plt.plot(
    arm_quad.pts + berry.logit_p1,
    uncon_pdf_db,
    color="k",
    linestyle="dotted",
    linewidth=3,
    label="Fisher Conjugation",
)
plt.plot(
    arm_quad.pts + berry.logit_p1,
    uncon_pdf_gaussian,
    color="k",
    linestyle="dashed",
    linewidth=3,
    label="Gaussian INLA",
)
plt.plot(
    arm_quad.pts + berry.logit_p1,
    uncon_pdf_interp,
    color="k",
    linestyle="solid",
    linewidth=3,
    label="Full INLA",
)
plt.hist(
    mcmc_results["x"][0]["theta"][0, :, 0] + berry.logit_p1,
    bins=np.linspace(-15, 2, 300),
    color="red",
    density=True,
    label="MCMC",
)
plt.xlim([-7, 0])
plt.ylim([0, 0.35])
plt.legend(fontsize=10)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\mathbb{P}(\theta_0 | Y)$")
plt.savefig('inla_compare.png', dpi=300, bbox_inches='tight')
plt.show()

```

```python
plt.plot(
    arm_quad.pts + berry.logit_p1,
    uncon_pdf_db,
    color="k",
    linestyle="dotted",
    linewidth=3,
    label="Fisher Conjugation",
)
plt.plot(
    arm_quad.pts + berry.logit_p1,
    uncon_pdf_gaussian,
    color="k",
    linestyle="dashed",
    linewidth=3,
    label="Gaussian INLA",
)
plt.plot(
    arm_quad.pts + berry.logit_p1,
    uncon_pdf_interp,
    color="k",
    linestyle="solid",
    linewidth=3,
    label="Full INLA",
)
plt.hist(
    mcmc_results["x"][0]["theta"][0, :, 0] + berry.logit_p1,
    bins=np.linspace(-15, 2, 300),
    color="red",
    density=True,
    label="MCMC",
)
plt.xlim([-2.0, 0.25])
plt.ylim([0, 2.5])
plt.legend(fontsize=10)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\mathbb{P}(\theta_0 | Y)$")
plt.show()
```
