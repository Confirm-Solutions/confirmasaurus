---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('confirm')
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import inlaw
import inlaw.berry as berry
import inlaw.quad as quad
import numpy as np
import jax.numpy as jnp
import jax
import time
import inlaw.inla as inla
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from functools import partial
from itertools import combinations

import lewis
import batch
```

# Lei Example


The following description is a clinical trial design using a Bayesian model with early-stopping rules for futility or efficacy of a drug.
This design was explicitly requested to be studied by an FDA member (Lei) in the CID team.


> The following is a randomized, double-blind, placebo-controlled two-stage adaptive design intended to identify an optimal treatment regimen 
> from three possible regimens (for example, different dosages or different combinations of agents) and 
> to assess the efficacy of that regimen with respect to a primary binary response endpoint measured at month 6.
> 
> In Stage 1, one of four experimental regimens will be selected, or the trial will stop for futility. 
> In this stage, a minimum of 200 and a maximum of 400 will be randomized 1:1:1:1 to one of the three experimental arms or one placebo arm. 
> Interim analyses will be conducted after 200, 300 and 400 subjects have been enrolled to select the best experimental regimen and to potentially stop 
> the trial for futility. 
> If an experimental regimen is dropped for futility at an interim analysis, 
> the next 100 subjects to be randomized will be allocated equally among the remaining arms in the study. 
> At each of these three analysis time points (N = 200, 300, 400), 
> the probabilities of being the best regimen (PrBest) and predictive probability of success (PPS) 
> are calculated for each experimental regimen using a Bayesian approach, 
> and the trial will either stop for futility, 
> continue to the next interim analysis, 
> or proceed to Stage 2 depending on the results of these PrBest and PPS calculations.
> 
> In Stage 2, a minimum of 200 and a maximum of 400 additional subjects will be randomized 1:1 to the chosen regimen or placebo. 
> The two groups (pooling both Stage 1 and Stage 2 subjects) will be compared for efficacy and futility assessment at an interim analysis 
> after 200 subjects have been enrolled in Stage 2, 
> and for efficacy at a final analysis after 400 subjects have been enrolled in Stage 2 and fully followed-up for response. 
> The study may be stopped for futility or efficacy based on PPS at the interim analysis. 
> If the study continues to the final analysis, 
> the posterior distribution of the difference in response rates between placebo and the chosen experimental arm 
> will be evaluated against a pre-specified decision criterion.
> 
> - Scenario 1: interim analyses are based on available data on the primary endpoint (measured at month 6)
> - Scenario 2: interim analyses are based on available data on a secondary endpoint (measured at month 3) 


This notebook breaks down and discusses the components of the trial.


## Model


The notation is as follows:


- $y \in \mathbb{N}^d$: Binomial responses.
- $p \in [0,1]^d$: probability parameter to the Binomial distribution.
- $n \in \mathbb{N}^d$: size parameter to the Binomial distribution.
- $q \in [0,1]^d$: base probability value to offset $p$.
- $\theta \in \R^d$: logit parameter that determines $p$.
- $\mu \in \mathbb{R}$: shared mean parameter among $\theta_i$.
- $\sigma^2 \in \mathbb{R}_+$: shared variance parameter among $\theta_i$.
- $\mu_0, \sigma_0^2, \alpha_0, \beta_0 \in \mathbb{R}$: hyper-parameters.


The Bayesian model is described below:
\begin{align*}
y_i | p_i &\sim \mathrm{Binom}(n_i, p_i) \quad i = 1,\ldots, d \\
p_i &= {\sf expit}(\theta_i + \mathrm{logit}(q_i) ) \quad i = 1,\ldots, d \\
\theta | \mu, \sigma^2 &\sim \mathcal{N}(\mu \mathbb{1}, \sigma^2 I) \\
\mu &\sim \mathcal{N}(\mu_0, \sigma_0^2) \\
\sigma^2 &\sim \Gamma^{-1}(\alpha_0, \beta_0) \\
\end{align*}

We note in passing that the model can be collapsed along $\mu$ to get:
\begin{align*}
y_i | p_i &\sim \mathrm{Binom}(n_i, p_i) \quad i = 1,\ldots, d \\
p_i &= {\sf expit}(\theta_i + \mathrm{logit}(q_i) ) \quad i = 1,\ldots, d \\
\theta | \sigma^2 &\sim \mathcal{N}(\mu_0 \mathbb{1}, \sigma^2 I + \sigma_0^2 \mathbb{1} \mathbb{1}^\top) \\
\sigma^2 &\sim \Gamma^{-1}(\alpha_0, \beta_0) \\
\end{align*}



## Probability of Best Arm


The first quantity of interest is probability of best (treatment) arm.
Concretely, letting $i = 1$ denote the control arm, we wish to compute for each $1 < i \leq d$:
\begin{align*}
\mathbb{P}(p_i > \max\limits_{j \neq i} p_j | y, n)
&=
\int \mathbb{P}(p_i > \max\limits_{j \neq i} p_j | y, n, \sigma^2) p(\sigma^2 | y, n) \, d\sigma^2
\\&=
\int \mathbb{P}(\theta_i + c_i > \max\limits_{j \neq i} (\theta_j + c_j) | y, n, \sigma^2) p(\sigma^2 | y, n) \, d\sigma^2
\end{align*}
where $c = \mathrm{logit}(q)$.
We can approximate this quantity by estimating the two integrand terms separately. 
By approximating $\theta_i | y, n$ as normal, the first integrand term can be estimated by Monte Carlo.
The second term can be estimated by INLA.

```python
def pr_normal_best(mean, cov, key, n_sims):
    '''
    Estimates P[X_i > max_{j != i} X_j] where X ~ N(mean, cov) via sampling.
    '''
    out_shape = (n_sims, *mean.shape[:-1])
    sims = jax.random.multivariate_normal(key, mean, cov, shape=out_shape)
    order = jnp.arange(1, mean.shape[-1])
    compute_pr_best_all = jax.vmap(lambda i: jnp.mean(jnp.argmax(sims, axis=-1) == i, axis=0))
    return compute_pr_best_all(order)
```

```python
d = 4
mean = jnp.array([2, 2, 2, 5])
cov = jnp.eye(d)
key = jax.random.PRNGKey(0)
n_sims = 100000
jax.jit(pr_normal_best, static_argnums=(3,))(mean, cov, key, n_sims=n_sims)
```

Next, we perform INLA to estimate $p(\sigma^2 | y, n)$ on a grid of values for $\sigma^2$.

```python
sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3)
sig2_rule_ops = berry.optimized(sig2_rule.pts, n_arms=4).config(
    opt_tol=1e-3
)
```

```python
def posterior_sigma_sq(data, sig2_rule, sig2_rule_ops):
    n_arms, _ = data.shape
    sig2 = sig2_rule.pts
    n_sig2 = sig2.shape[0]
    p_pinned = dict(sig2=sig2, theta=None)

    f = sig2_rule_ops.laplace_logpost
    logpost, x_max, hess, iters = f(
        np.zeros((n_sig2, n_arms)), p_pinned, data
    )
    post = inla.exp_and_normalize(
            logpost, sig2_rule.wts, axis=-1)

    return post, x_max, hess, iters 
```

```python
dtype = jnp.float64
N = 1
data = berry.figure2_data(N).astype(dtype)[0]
n_arms, _ = data.shape
posterior_sigma_sq_jit = jax.jit(lambda data: posterior_sigma_sq(data, sig2_rule, sig2_rule_ops))
post, _, hess, _ = posterior_sigma_sq_jit(data)
```

Putting the two pieces together, we have the following function to compute the probability of best treatment arm.

```python
def pr_best(data, sig2_rule, sig2_rule_ops, key, n_sims):
    n_arms, _ = data.shape
    post, x_max, hess, _ = posterior_sigma_sq(data, sig2_rule, sig2_rule_ops) 
    mean = x_max
    hess_fn = jax.vmap(lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1]))
    prec = -hess_fn(hess) # (n_sigs, n_arms, n_arms)
    cov = jnp.linalg.inv(prec)
    pr_normal_best_out = pr_normal_best(mean, cov, key=key, n_sims=n_sims)
    return jnp.matmul(pr_normal_best_out, post * sig2_rule.wts)
```

```python
n_sims = 13
out = pr_best(data, sig2_rule, sig2_rule_ops, key, n_sims)
out
```

## Phase III Final Analysis

\begin{align*}
\mathbb{P}(\theta_i - \theta_0 < t | y, n) < 0.1
\end{align*}

\begin{align*}
\mathbb{P}(\theta_i - \theta_0 < t | y, n)
&=
\mathbb{P}(q_1^\top \theta < t | y,n)
=
\int \mathbb{P}(q_1^\top \theta < t | y, n, \sigma^2) p(\sigma^2 | y, n) \, d\sigma^2
\\&=
\int \mathbb{P}(q_1^\top \theta < t | y, n, \sigma^2) p(\sigma^2 | y, n) \, d\sigma^2
\\
q_1^\top \theta | y, n, \sigma^2 &\sim \mathcal{N}(q_1^\top \theta^*, -q_1^\top (H\log p(\theta^*, y, \sigma^2))^{-1} q_1)
\end{align*}

```python
posterior_difference_threshold = 0.2
rejection_threshold = 0.1
```

```python
def posterior_difference(data, arm, sig2_rule, sig2_rule_ops, thresh):
    n_arms, _ = data.shape
    post, x_max, hess, _ = posterior_sigma_sq(data, sig2_rule, sig2_rule_ops)
    hess_fn = jax.vmap(lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1]))
    prec = -hess_fn(hess) # (n_sigs, n_arms, n_arms)
    order = jnp.arange(0, n_arms)
    q1 = jnp.where(order == 0, -1, 0)
    q1 = jnp.where(order == arm, 1, q1)
    loc = x_max @ q1
    scale = jnp.linalg.solve(prec, q1[None,:]) @ q1
    normal_term = jax.scipy.stats.norm.cdf(thresh, loc=loc, scale=scale)
    post_weighted = sig2_rule.wts * post
    out = normal_term @ post_weighted
    return out

posterior_difference(data, 1, sig2_rule, sig2_rule_ops, posterior_difference_threshold)
```

## Posterior Probability of Success

The next quantity we need to compute is the posterior probability of success (PPS).
For convenience of implementation, we will take this to mean the following:
let $y, n$ denote the currently observed data
and $A_i = \{ \text{Phase III rejects using treatment arm i} \}$.
Then, we wish to compute
\begin{align*}
\mathbb{P}(A_i | y, n)
\end{align*}
Expanding the quantity,
\begin{align*}
\mathbb{P}(A_i | y, n) &=
\int \mathbb{P}(A_i | y, n, \theta_i, \theta_0) p(\theta_0, \theta_i | y, n) \, d\theta_i d\theta_0 \\&=
\int \mathbb{P}(A_i | y, n, \theta_i, \theta_0) p(\theta_0, \theta_i | y, n) \, d\theta_i d\theta_0
\end{align*}

Once we have an estimate for $p(\theta_0, \theta_i | y, n)$, 
we can use 2-D quadrature to numerically integrate the integrand.
Similar to computing the probability of best arm,
\begin{align*}
p(\theta_0, \theta_i | y, n)
&=
\int p(\theta_0, \theta_i | y, n, \sigma^2) p(\sigma^2 | y, n) \, d\sigma^2
\end{align*}
We will use the Gaussian approximation for $p(\theta_0, \theta_i | y, n, \sigma^2)$
and use INLA to estimate $p(\sigma^2 | y, n)$.

\begin{align*}
p(\theta | y, n, \sigma^2)
\approx
\mathcal{N}(\theta^*, -(H\log p(\theta^*, y, \sigma^2))^{-1})
\\
\implies
p(\theta_0, \theta_i | y, n, \sigma^2)
\approx
\mathcal{N}(\theta^*_{[0,i]}, -(H\log p(\theta^*, y, \sigma^2))^{-1}_{[0,i], [0,i]})
\end{align*}

```python
# input parameters
n_Ai_sims = 1000
p = jnp.full(n_arms, 0.5)
n_stage_2 = 100
pps_threshold_lower = 0.1
pps_threshold_upper = 0.9
posterior_difference_threshold = 0.1
rejection_threshold = 0.1

subset = jnp.array([0, 1])
non_futile_idx = np.zeros(n_arms)
non_futile_idx[subset] = 1
non_futile_idx = jnp.array(non_futile_idx)

# create a dense grid of sig2 values
n_sig2 = 100
sig2_grid = 10**jnp.linspace(-6, 3, n_sig2)
dsig2_grid = jnp.diff(sig2_grid)
sig2_grid_ops = berry.optimized(sig2_grid, n_arms=data.shape[0]).config(
    opt_tol=1e-3
)

_, key = jax.random.split(key)
```

```python
def pr_Ai(
    data, p, key, best_arm, non_futile_idx, 
    sig2_rule, sig2_rule_ops,
    sig2_grid, sig2_grid_ops, dsig2_grid,
):
    n_arms, _ = data.shape

    # compute p(sig2 | y, n), mode, hessian
    p_pinned = dict(sig2=sig2_grid, theta=None)
    logpost, x_max, hess, _ = jax.jit(sig2_grid_ops.laplace_logpost)(
        np.zeros((len(sig2_grid), n_arms)), p_pinned, data
    )
    max_logpost = jnp.max(logpost)
    max_post = jnp.exp(max_logpost)
    post = jnp.exp(logpost - max_logpost) * max_post

    # sample sigma^2 | y, n
    dFx = post[:-1] * dsig2_grid
    Fx = jnp.cumsum(dFx)
    Fx /= Fx[-1]
    _, key = jax.random.split(key)
    unifs = jax.random.uniform(key=key, shape=(n_Ai_sims,))
    i_star = jnp.searchsorted(Fx, unifs)

    # sample theta | y, n, sigma^2
    mean = x_max[i_star+1]
    hess_fn = jax.vmap(
        lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1])
    )
    prec = -hess_fn(hess)
    cov = jnp.linalg.inv(prec)[i_star+1]
    _, key = jax.random.split(key)
    theta = jax.random.multivariate_normal(
        key=key, mean=mean, cov=cov,
    )
    p_samples = jax.scipy.special.expit(theta)

    # estimate P(A_i | y, n, theta_0, theta_i)

    def simulate_Ai(data, best_arm, diff_thresh, rej_thresh, non_futile_idx, key, p):
        # add n_stage_2 number of patients to each
        # of the control and selected treatment arms.
        n_new = jnp.where(non_futile_idx, n_stage_2, 0)
        y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)

        # pool outcomes for each arm
        data = data + jnp.stack((y_new, n_new), axis=-1)

        return posterior_difference(data, best_arm, sig2_rule, sig2_rule_ops, diff_thresh) < rej_thresh

    simulate_Ai_vmapped = jax.vmap(
        simulate_Ai, in_axes=(None, None, None, None, None, 0, 0)
    )
    keys = jax.random.split(key, num=p_samples.shape[0])
    Ai_indicators = simulate_Ai_vmapped(
        data,
        best_arm,
        posterior_difference_threshold,
        rejection_threshold,
        non_futile_idx,
        keys,
        p_samples,
    )
    out = jnp.mean(Ai_indicators)
    return out

```

```python
%%time
jax.jit(lambda data, p, key, best_arm, non_futile_idx:
    pr_Ai(data, p, key, best_arm, non_futile_idx, 
          sig2_rule, sig2_rule_ops, sig2_grid, sig2_grid_ops, dsig2_grid),
    static_argnums=(3,))(data, p, key, 1, non_futile_idx)
```

```python
# Sampling based on pdf values and linearly interpolating
n_sims = 1000
n_unifs = 1000
key = jax.random.PRNGKey(2)

#x = jnp.linspace(-3, 3, num=n_sims)
#px_orig = 0.5*jax.scipy.stats.norm.pdf(x, -1, 0.5) + 0.5*jax.scipy.stats.norm.pdf(x, 1, 0.5)

#x = jnp.linspace(0, 10, num=n_sims)
#px_orig = jax.scipy.stats.gamma.pdf(x, 10)

x = jnp.linspace(0, 1, num=n_sims)
px_orig = jax.scipy.stats.beta.pdf(x, 4, 2)

px = 2 * px_orig
dx = jnp.diff(x)
dFx = px[:-1] * dx
Fx = jnp.cumsum(dFx)
Fx /= Fx[-1]
_, key = jax.random.split(key)
unifs = jax.random.uniform(key=key, shape=(n_unifs,))
i_star = jnp.searchsorted(Fx, unifs)

# point mass approx
#samples = x[i_star+1]

# constant approx
#samples = x[i_star+1] - (Fx[i_star] - unifs) / px[i_star]

# linear approx
a = 0.5 * (px[i_star+1] - px[i_star]) / dx[i_star]
b = px[i_star]
c = Fx[i_star] - unifs - px[i_star] * dx[i_star] - a * dx[i_star]**2
discr = jnp.sqrt(jnp.maximum(b**2 - 4*a*c, 0))
quad_solve = jnp.where(jnp.abs(a) < 1e-8, -c/b, (-b + discr) / (2*a))
samples = x[i_star] + quad_solve

#plt.plot(x[1:], Fx)
#plt.plot(x[1:], jax.scipy.stats.norm.cdf(x[1:]))
plt.hist(x[i_star+1], density=True, alpha=0.5)
plt.hist(samples, density = True, alpha=0.5)
plt.plot(x, px_orig)
plt.show()
```

## Design Implementation

```python
%%time
key = jax.random.PRNGKey(0)
params = {
    "n_arms" : 3,
    "n_stage_1" : 10,
    "n_stage_2" : 10,
    "n_interims" : 3,
    "n_add_per_interim" : 4,
    "futility_threshold" : 0.1,
    "pps_threshold_lower" : 0.1,
    "pps_threshold_upper" : 0.9,
    "posterior_difference_threshold" : 0.05,
    "rejection_threshold" : 0.05,
    "n_pr_sims" : 100,
    "n_sig2_sim" : 20,
    "batch_size" : 2**10,
}
lei_obj = lewis.Lewis45(**params)
p = jnp.array([0.5] * params['n_arms'])
#grid_points = jnp.array([p] * n_grid_points)
```

```python
@partial(jax.jit, static_argnums=(3,))
def generate_data(n, p, key, n_sims=-1):
    if n_sims == -1:
        y = dist.Binomial(n, p).sample(key)
        data = jnp.stack((y, n), axis=-1)
    else:
        y = dist.Binomial(n, p).sample(key, (n_sims,))
        n_full = jnp.full_like(y, n)
        data = jnp.stack((y, n_full), axis=-1)
    return data
```

```python
%%time 
n = jnp.array([params['n_stage_1']]*params['n_arms']) 
n_data = 10
data = generate_data(n, p, key, n_data)
sim_keys = jax.random.split(key, num=n_data)
```

```python
%%time
#rejections = jax.jit(lei_obj.single_sim)(p, key)
#rejections = jax.jit(lei_obj.simulate_point)(p, keys)
#rejections = jax.jit(lei_obj.simulate)(grid_points, keys)
#jnp.mean(rejections, axis=-1)
```

```python
%%time
pd = lei_obj.posterior_difference_table__(batch_size=params['batch_size'])
pd.shape
```

```python
%%time
pr_best, pps = lei_obj.pr_best_pps_table__(key, batch_size=params['batch_size'])
pr_best.shape, pps.shape
```
