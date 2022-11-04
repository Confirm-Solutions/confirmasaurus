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

# Conditional Laplace INLA

Let's first just set up our environment. The dataset is the one from Figure 2 from Berry 2013.

```python
import berrylib.util as util

util.setup_nb()

import pickle
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

import outlaw
import berrylib.mcmc as mcmc

data = outlaw.berry_model.figure2_data(N=1)
```

And load up some MCMC results, both for the full problem and conditional on various values of $\sigma^2$.

```python
sig2_n = 15
sig2_rule = outlaw.quad.log_gauss_rule(N=sig2_n, a=1e-6, b=1e3)

load = True
if load:
    with open("conditional_mcmc.pkl", "rb") as f:
        mcmc_results, mcmc_arm_marg = pickle.load(f)
else:
    mcmc_results = []
    for sig_idx in range(fi.sigma2_n):
        mcmc_results.append(
            mcmc.mcmc_berry(
                np.stack((y[-1:], n[-1:]), axis=-1),
                n_samples=200000,
                sigma2_val=sig2_rule.pts[sig_idx],
            )
        )
    mcmc_arm_marg = mcmc.mcmc_berry(
        np.stack((y[-1:], n[-1:]), axis=-1),
        n_samples=200000,
    )
    with open("conditional_mcmc.pkl", "wb") as f:
        pickle.dump((mcmc_results, mcmc_arm_marg), f)
```

Then we calculate the hyperparameter posteriors in the variable `post`. 

```python
fl = outlaww.FullLaplaceoutlaww.berry_model.berry_model(4), "sig2", np.zeros((4, 2))
p_pinned = dict(sig2=sig2_rule.pts, theta=None)
logpost, x_max, hess, iters = fl(p_pinned, data)
post = outlaw.inla.exp_and_normalize(logpost, sig2_rule.wts, axis=1)
```

## Choosing points for INLA

One of the difficult issues with the more accurate versions of INLA is that we need to choose a grid of points at which to evaluate the latent variable marginals.

Ideally, this choice is automatic.


### Gauss-Hermite quadrature works nicely for integrating well-behaved densities from $-\infty$ to $\infty$

**Gauss-Hermite quadrature**: is perfect for this use case since it's designed to integrate functions over the entire real domain that look like Gaussians multiplied by well-behaved polynomial-ish functions:

$$
\int_{-\infty}^{\infty} f(x) e^{-x^2} dx
$$

**Scaling Gauss-Hermite quadrature**: 
One way of thinking about this is that GH quadrature is good at integrating the unit normal distribution. Via change of variables, we can transform other normal distributions into the unit and integrate them with GH quadrature too.  

The figure below demonstrates that this results in much faster convergence and more accurate integrals. If the scale = 2.0 (the true standard deviation), then convergence is fastest. But scales that are slightly different are okay too. If the scale is very different, then convergence is quite slow!

Since we know the hessian at the mode, we can give GH quadrature a good guess of the standard deviation and get nice integrals.

```python
def f(x):
    return np.sin(x) ** 2 * np.exp(-((x / 2) ** 2))


def gauss_I(n, scale=1.0):
    gr = util.gauss_rule(n, -10 * scale, 10 * scale)
    gp, gw = gr.pts, gr.wts
    y = f(gp)
    return np.sum(y * gw)
```

```python
def herm_I(n, scale=1.0):
    hr = outlaw.quad.gauss_herm_rule(n, center=0, scale=scale)
    return np.sum(hr.wts * f(hr.pts))
```

The accuracy is best when the scale matches the standard deviation of the density, but it still works well even if the standard deviation is somewhat off.

```python
exact = herm_I(100, scale=2.0)
gerr = []
scales = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
herr = [[] for i in range(len(scales))]
ns = range(4, 100, 2)
for n in ns:
    gerr.append(np.abs(gauss_I(n, scale=1.0) - exact))
    for i, scale in enumerate(scales):
        herr[i].append(np.abs(herm_I(n, scale=scale) - exact))
# Plot comparing gauss quadrature and hermite quadrature
plt.figure(figsize=(8, 8))
plt.plot(ns, np.log10(gerr), "b-", label="Gauss", linewidth=3.0)
for i in range(len(herr)):
    plt.plot(
        ns, np.log10(herr[i]), label=f"Hermite scale={scales[i]:.2f}", linewidth=3.0
    )
plt.legend()
plt.xlabel("Number of quadrature points")
plt.ylabel("Error")
plt.title("Error in quadrature")
plt.show()
```

## Running conditional INLA with Gauss-Hermite


In this first section, we're going to compute the grid of points separately for each value of the hyperparameter $\sigma^2$. This results is really clean plots of the conditional densities: $p(\theta_i | \sigma^2, y)$

But, it has the downside of making it hard to produce a plot of the non-conditional arm marginal: $p(\theta_i | y)$
The problem is that we need to integrate over $\sigma^2$ but our different conditional densities aren't on the same $\theta_i$ grid.

Obviously, I've chosen a particularly bad example here to emphasize the differences. Note the *very skewed and broad* distributions for large $\sigma^2$

```python
arm_idx = 0
cond_inla_f = outlaw.inla.build_conditional_inla(fl.log_joint_single, fl.spec)
cx, wts = outlaw.inla.gauss_hermite_grid(x_max, hess, arm_idx, n=25)
lp = cond_inla_f(x_max, p_pinned, data, hess, cx, arm_idx)
arm_marg = outlaww.inla.exp_and_normalize(lp, wts, axis=0)
```

```python
plt.figure(figsize=(8, 12), constrained_layout=True)
for j, sig_idx in enumerate(range(sig2_n)[::3]):
    mcmc_arm = mcmc_results[sig_idx]["x"][0]["theta"][0, :, arm_idx]
    mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, cx[:, 0, sig_idx], wts[:, 0, sig_idx])

    plt.subplot(3, 2, 1 + j)
    plt.title(f"$\sigma^2$ = {sig2_rule.pts[sig_idx]:5.2e}")
    plt.plot(cx[:, 0, sig_idx], mcmc_p_ti_g_y, "b-")
    plt.plot(cx[:, 0, sig_idx], arm_marg[:, 0, sig_idx], "r-o", linewidth=2.0)
    plt.ylabel(r"$p(\theta_0 | \sigma^2, y)$")
    plt.xlabel(r"$\theta_0$")
    window = np.quantile(mcmc_arm, [0.001, 0.999])
    # plt.hist(mcmc_arm, bins=np.linspace(*window, 40), density=True)
plt.show()
```

```python
mcmc_pdfs = []
for j, sig_idx in enumerate(range(sig2_n)):
    mcmc_arm = mcmc_results[sig_idx]["x"][0]["theta"][0, :, arm_idx]
    mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, cx[:, 0, sig_idx], wts[:, 0, sig_idx])
    mcmc_pdfs.append(mcmc_p_ti_g_y)

mcmc_pdfs = np.array(mcmc_pdfs).T.copy()
test_data = np.stack((cx[:, 0, :], mcmc_pdfs), axis=-1)
np.save("test_conditional_inla.npy", test_data)
```

Because the grids are different for each value of $\sigma^2$, it's not very easy to integrate across $\sigma^2$. It still would be possible if we did some sort of interpolation, but it's not a trivial thing.


### A single set of points for all $\sigma^2$

If instead, we choose a single grid of $\theta_i$ values for all $\sigma^2$, the results are not nearly as pretty.

BUT, the advantage is that computing the non-conditional marginal distribution is a very simple integration. 


```python
x_sigma2 = -np.diagonal(np.linalg.inv(hess), axis1=2, axis2=3)
mu_arm = np.sum(post[..., None] * x_max * sig2_rule.wts[None, :, None], axis=1)
var_arm = np.sum(
    post[..., None]
    * ((x_max - mu_arm[:, None, :]) ** 2 + x_sigma2)
    * sig2_rule.wts[None, :, None],
    axis=1,
)
mu_arm, var_arm
sd_arm = np.sqrt(var_arm)
```

```python
hg_rule = outlaw.quad.gauss_herm_rule(70)
hg_pts, hg_wts = hg_rule.pts, hg_rule.wts
cx = np.tile(
    mu_arm[:, None, arm_idx] + sd_arm[:, None, arm_idx] * hg_pts[:, None, None],
    (1, 1, sig2_n),
)
lp = cond_inla_f(x_max, p_pinned, data, hess, cx, arm_idx)
wts = np.tile(sd_arm[:, None, arm_idx] * hg_wts[:, None, None], (1, 1, sig2_n))
arm_marg = outlaw.inla.exp_and_normalize(lp, wts, axis=0)
```

```python
plt.figure(figsize=(8, 12), constrained_layout=True)
for j, sig_idx in enumerate(range(sig2_n)[::3]):
    mcmc_arm = mcmc_results[sig_idx]["x"][0]["theta"][0, :, arm_idx]
    mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, cx[:, 0, sig_idx], wts[:, 0, sig_idx])

    plt.subplot(3, 2, 1 + j)
    plt.title(f"$\sigma^2$ = {sig2_rule.pts[sig_idx]:5.2e}")
    plt.plot(cx[:, 0, sig_idx], mcmc_p_ti_g_y, "b-")
    plt.plot(cx[:, 0, sig_idx], arm_marg[:, 0, sig_idx], "r-o", linewidth=2.0)
    plt.ylabel(r"$p(\theta_0 | \sigma^2, y)$")
    plt.xlabel(r"$\theta_0$")
    window = np.quantile(mcmc_arm, [0.001, 0.999])
    plt.hist(mcmc_arm, bins=np.linspace(*window, 40), density=True)
plt.show()
```

```python
mcmc_arm = mcmc_arm_marg["x"][0]["theta"][0, :, arm_idx]
int_arm_marg = np.sum(arm_marg * post[None] * sig2_rule.wts[None, None], axis=2)
window = np.quantile(mcmc_arm, [0.01, 0.999])
plt.hist(mcmc_arm, bins=np.linspace(*window, 100), density=True)
plt.plot(cx[:, 0, 0], int_arm_marg, "k-", linewidth=2.0)
plt.xlim(window)
plt.ylabel(r"$p(\theta_0 | y)$")
plt.xlabel(r"$\theta_0$")
plt.show()
```

### So what should we do with the arm marginals?

The two options:
1. Choose $\theta_i$ grids conditional on $\sigma^2$. This produces much more accurate conditionals but makes it hard to produce a final arm marginal.
2. Choose $\theta_i$ grids without regard to $\sigma^2$. Producing a final arm marginal is easier, but the conditional distributions are ugly!

After writing all this, I think the choice is fairly obvious. What is our main goal here? Normally, we just want to compute an exceedance! And that exceedance will be best computed in the conditional world before being integrated. So, having excellent conditionals is more important than making it easy to produce unconditional marginals. 

<!-- #region -->
## Other stuff that's just coincidentally still in this notebook: Jensen-Shannon divergence

[Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)

https://stats.stackexchange.com/questions/6907/an-adaptation-of-the-kullback-leibler-distance/6937#6937


The symmetric KL divergence doesn't behave well when there are zeros in either distribution. The JS divergence solves this issue by comparing to the midpoint distribution.

More generally, the issue we are faced with is computing a norm of distributions. so something like the L2 norm of the difference of the functions would be fine. But the more probabalistic/informational divergence metrics are probably better.

One way we can avoid this entirely is to move to evaluating error in terms of the final output, the exceedance. In many cases, all we are going to care about is the exceedance prob for some threshold. This is a scalar (or vector with multiple arms) and would make error tolerances more concrete and understandable. 
<!-- #endregion -->

```python
import scipy.special


def js_div(x, y, wts):
    R = 0.5 * (x + y)
    a = np.sum(wts * scipy.special.rel_entr(x, R))
    b = np.sum(wts * scipy.special.rel_entr(y, R))
    return 0.5 * (a + b)
```

```python
import pandas as pd

df = pd.DataFrame(dict(sig2=fi.sigma2_rule.pts))
df["JS_G_M"] = 0
df["JS_L_M"] = 0
for i, sig_idx in enumerate(range(fi.sigma2_n)):
    mcmc_arm = mcmc_results[sig_idx]["x"][0]["theta"][0, :, arm_idx]
    mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, t_i.pts, t_i.wts)
    gaussian = scipy.stats.norm.pdf(
        t_i.pts, theta_max[-1, sig_idx, 0], theta_sigma[-1, sig_idx, 0]
    )
    df.at[i, "JS_G_M"] = js_div(mcmc_p_ti_g_y, gaussian, t_i.wts)
    df.at[i, "JS_L_M"] = js_div(mcmc_p_ti_g_y, laplace[sig_idx], t_i.wts)
df
```

```python

```

```python
plt.plot(np.log10(df["sig2"]), np.log10(df["JS_G_M"]), "b-", label="Gaussian")
plt.plot(np.log10(df["sig2"]), np.log10(df["JS_L_M"]), "k-", label="Full laplace")
plt.legend()
plt.show()
```

## Yeah, you should probably stop here... The stuff above is cleaned up. Below is just notes to self.


### adaptive quadrature rule.

this will get far enough out from the mode, but is more complicated and possibly slower. not sure!

```python
mu_arm = x_max
sd_arm = np.sqrt(-np.diagonal(np.linalg.inv(hess), axis1=2, axis2=3))
```

```python
def eval_cx(cx):
    return cond_inla_f(x_max, p_pinned, data, hess, cx, arm_idx)


cx_all = [mu_arm[None, ..., arm_idx]]
arm_marg_all = [eval_cx(mu_arm[None, ..., arm_idx])]
for i in range(8):
    a = i
    b = i + 1

    domain = np.linspace(a, b, 5)[1:]
    cx = mu_arm[None, ..., arm_idx] + sd_arm[None, ..., arm_idx] * domain[:, None, None]
    lp = eval_cx(cx)

    cx_all.append(cx)
    arm_marg_all.append(lp)
```

```python
cx_all = np.concatenate(cx_all, axis=0)
arm_marg_all = np.concatenate(arm_marg_all, axis=0)
```

```python
order = np.argsort(cx_all[:, 0, 0])
cx_all = cx_all[order]
arm_marg_all = arm_marg_all[order]
a = cx_all[0]
b = cx_all[-1]
wts = np.tile(((b - a) / cx_all.shape[0])[None], (cx_all.shape[0], 1, 1))
arm_marg = outlaw.inla.exp_and_normalize(arm_marg_all, wts, axis=0)
```

```python
plt.figure(figsize=(12, 4))
for j, sig_idx in enumerate(range(fi.sigma2_n)[::3]):
    mcmc_arm = mcmc_results[sig_idx]["x"][0]["theta"][0, :, arm_idx]
    mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, cx_all[:, 0, sig_idx], wts[:, 0, sig_idx])

    plt.subplot(1, 5, 1 + j)
    plt.title(f"$\sigma^2$ = {fi.sigma2_rule.pts[sig_idx]:5.2e}")
    plt.plot(cx_all[:, 0, sig_idx], mcmc_p_ti_g_y, "b-")
    # plt.plot(t_i.pts, arm_marg[:, 0, sig_idx], 'r-.', linewidth=2.0)
    plt.plot(cx_all[:, 0, sig_idx], arm_marg[:, 0, sig_idx], "r-.", linewidth=2.0)
plt.show()
```

### pre-specified fixed quad rule.

this is slow and silly, but works.

```python
arm_idx = 0
t_i = util.simpson_rule(101, -15, 2)
p_pinned = dict(sig2=fi.sigma2_rule.pts, theta=None)
cond_inla_f = outlaw.inla.build_conditional_inla(fl.log_joint_single, fl.spec)
cx = np.tile(t_i.pts[:, None, None], (1, x_max.shape[0], x_max.shape[1]))
lp = cond_inla_f(x_max, p_pinned, data, hess, cx, arm_idx)
arm_marg = outlaw.inla.exp_density(lp, t_i.wts[:, None, None], axis=0)
```

```python
plt.figure(figsize=(12, 4))
for j, sig_idx in enumerate(range(fi.sigma2_n)[5::2]):
    mcmc_arm = mcmc_results[sig_idx]["x"][0]["theta"][0, :, arm_idx]
    mcmc_p_ti_g_y = mcmc.calc_pdf(mcmc_arm, t_i.pts, t_i.wts)

    plt.subplot(1, 5, 1 + j)
    plt.title(f"$\sigma^2$ = {fi.sigma2_rule.pts[sig_idx]:5.2e}")
    plt.plot(t_i.pts, mcmc_p_ti_g_y, "b-")
    plt.plot(t_i.pts, arm_marg[:, 0, sig_idx], "r-.", linewidth=2.0)
    window = np.quantile(mcmc_arm, [0.0005, 0.9995])
    plt.hist(mcmc_arm, bins=np.linspace(*window, 31), density=True)
    plt.xlim(*window)
plt.show()
```
