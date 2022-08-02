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
import inlaw
import inlaw.berry as berry
import inlaw.quad as quad
import numpy as np
import jax.numpy as jnp
import jax
import time
import inlaw.inla as inla
import numpyro.distributions as dist
from functools import partial
```

```python
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

        my_timeit(N, jax.jit(f), iter=iter)

    custom_ops = berry.optimized(sig2, dtype=dtype).config(max_iter=10)
    bench_ops("custom berry", custom_ops)

    ad_ops = inla.from_log_joint(
        berry.log_joint(4), dict(sig2=np.array([np.nan]), theta=np.full(4, 0.0))
    ).config(max_iter=10)
    bench_ops("numpyro berry", ad_ops)
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
custom_ops = berry.optimized(sig2_rule.pts, n_arms=4).config(
    opt_tol=1e-3
)

```

```python
def posterior_sigma_sq(data, pts, wts):
    _, n_arms, _ = data.shape
    sig2 = pts
    n_sig2 = sig2.shape[0]
    p_pinned = dict(sig2=sig2, theta=None)

    f = jax.jit(jax.vmap(custom_ops.laplace_logpost, in_axes=(None, None, 0)))
    logpost, x_max, hess, iters = f(
        np.zeros((n_sig2, n_arms)), p_pinned, data
    )
    post = inla.exp_and_normalize(
            logpost, wts, axis=1)

    return post, x_max, hess, iters 
```

```python
dtype = jnp.float64
N = 1
data = berry.figure2_data(N).astype(dtype)[0]
post, _, hess, _ = jax.jit(posterior_sigma_sq)(data[None,:], sig2_rule.pts, sig2_rule.wts)
n_arms = 4
ff = jax.jit(jax.vmap(jax.vmap(
    lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1]))))
```

Putting the two pieces together, we have the following function to compute the probability of best treatment arm.

```python
def pr_best(data, pts, wts, key, n_sims):
    n_arms, _ = data.shape
    post, x_max, hess, _ = posterior_sigma_sq(data[None,:], pts, wts) 
    post = post[0]
    x_max = x_max[0]
    hess = (hess[0][0], hess[1][0])
    mean = x_max
    hess_fn = jax.vmap(lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1]))
    prec = -hess_fn(hess) # (n_sigs, n_arms, n_arms)
    cov = jnp.linalg.inv(prec)
    pr_normal_best_out = pr_normal_best(mean, cov, key=key, n_sims=n_sims)
    return jnp.matmul(pr_normal_best_out, post * wts)
```

```python
n_sims = 13
#out = jax.jit(pr_best, static_argnums=(4,))(data, sig2_rule.pts, sig2_rule.wts, key, n_sims)
out = pr_best(data, sig2_rule.pts, sig2_rule.wts, key, n_sims)
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
post, x_max, hess, _ = posterior_sigma_sq(data[None,:], sig2_rule.pts, sig2_rule.wts)
post = post[0]
post_weighted = sig2_rule.wts * post

x_max = x_max[0]
hess = (hess[0][0], hess[1][0])
hess_fn = jax.vmap(lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1]))
prec = -hess_fn(hess) # (n_sigs, n_arms, n_arms)

q1 = np.zeros(n_arms)
q1[0] = -1
q1[1] = 1
q1 = jnp.array(q1)
loc = x_max @ q1
scale = jnp.linalg.solve(prec, q1[None,:]) @ q1
normal_term = jax.scipy.stats.norm.cdf(posterior_difference_threshold, loc=loc, scale=scale)
posterior_difference = normal_term @ post_weighted

posterior_difference
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

Method 1:



## Design Implementation

```python
%%time
key = jax.random.PRNGKey(0)
```

```python
class Lewis45:
    def __init__(
        self,
        n_arms,
        n_stage_1,
        n_stage_2,
        n_interims,
        n_add_per_interim,
        futility_threshold,
        pps_threshold_lower,
        pps_threshold_upper,
        posterior_difference_threshold,
        rejection_threshold,
        sig2_rule = quad.log_gauss_rule(15, 1e-6, 1e3),
        n_pr_best_sims = 100,
        dtype = jnp.float64,
    ):
        """
        Constructs an object to run the Lei example.

        Parameters:
        -----------
        n_stage_1:      number of patients to enroll at stage 1 for each arm.
        n_interims:     number of interims.
        n_add_per_interim:      number of total patients to add per interim.
        futility_threshold:     probability cut-off to decide futility for treatment arms.
                                If P(arm_i best | data) < futility_threshold, declare arm_i as futile.
        n_stage_2:              number of patients to add for stage 2 for each arm.
        pps_threshold_lower:    threshold for checking futility: PPS < pps_threshold_lower <=> futility.
        pps_threshold_upper:    threshold for checking efficacy: PPs > pps_threshold_upper <=> efficacy.
        posterior_difference_threshold: threshold to compute posterior difference of selected arm p and control arm p.
        rejection_threshold:    threshold for rejection at the final analysis (if reached):
                                P(p_selected_treatment_arm - p_control_arm < posterior_difference_threshold | data) < rejection_threshold
                                <=> rejection.
        """
        self.n_arms = n_arms
        self.n_stage_1 = n_stage_1
        self.n_stage_2 = n_stage_2
        self.n_interims = n_interims
        self.n_add_per_interim = n_add_per_interim
        self.futility_threshold = futility_threshold
        self.pps_threshold_lower = pps_threshold_lower
        self.pps_threshold_upper = pps_threshold_upper
        self.posterior_difference_threshold = posterior_difference_threshold
        self.rejection_threshold = rejection_threshold
        self.n_pr_best_sims = n_pr_best_sims
        
        self.dtype = dtype

        self.sig2_rule = sig2_rule
        self.sig2_rule.pts = self.sig2_rule.pts.astype(self.dtype)
        self.sig2_rule.wts = self.sig2_rule.wts.astype(self.dtype)
        self.custom_ops = berry.optimized(self.sig2_rule.pts, n_arms=n_arms).config(
            opt_tol=1e-3
        )
        
    def posterior_sigma_sq(self, data):
        '''
        Computes the posterior of sigma^2 given data, p(sigma^2 | y)
        using INLA method.
        '''
        _, n_arms, _ = data.shape
        sig2 = self.sig2_rule.pts
        n_sig2 = sig2.shape[0]
        p_pinned = dict(sig2=sig2, theta=None)
        f = jax.vmap(self.custom_ops.laplace_logpost, in_axes=(None, None, 0))
        logpost, x_max, hess, iters = f(
            np.zeros((n_sig2, n_arms), dtype=self.dtype), p_pinned, data
        )
        post = inla.exp_and_normalize(logpost, self.sig2_rule.wts[None, :], axis=1)

        return post, x_max, hess, iters 

    @staticmethod
    def pr_normal_best(mean, cov, key, n_sims):
        '''
        Estimates P[X_i > max_{j != i} X_j] where X ~ N(mean, cov) via sampling.
        '''
        out_shape = (n_sims, *mean.shape[:-1])
        sims = jax.random.multivariate_normal(key, mean, cov, shape=out_shape)
        order = jnp.arange(1, mean.shape[-1])
        compute_pr_best_all = jax.vmap(lambda i: jnp.mean(jnp.argmax(sims, axis=-1) == i, axis=0))
        return compute_pr_best_all(order)

    def compute_pr_best(self, data, non_futile_idx):
        n_arms, _ = data.shape
        post, x_max, hess, _ = self.posterior_sigma_sq(data[None,:]) 
        post = post[0]
        x_max = x_max[0]
        hess = (hess[0][0], hess[1][0])
        mean = x_max
        hess_fn = jax.vmap(lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1]))
        prec = -hess_fn(hess) # (n_sigs, n_arms, n_arms)
        cov = jnp.linalg.inv(prec)
        pr_normal_best_out = Lewis45.pr_normal_best(mean, cov, key=key, n_sims=self.n_pr_best_sims)
        pr_best_out = jnp.matmul(pr_normal_best_out, post * self.sig2_rule.wts)
        return jnp.where(non_futile_idx == 0, jnp.nan, pr_best_out)
             
    @staticmethod
    def compute_pps(data):
        # TODO: fill in detail
        return 0.5

    def posterior_difference(self, data, arm, thresh):
        # TODO: p(sigma^2 | y) can be precomputed in a table?
        #n_arms, _ = data.shape
        #post, x_max, hess, _ = self.posterior_sigma_sq(data[None,:])
        #post = post[0]
        #x_max = x_max[0]
        #hess = (hess[0][0], hess[1][0])

        #post_weighted = self.sig2_rule.wts * post
        #hess_fn = jax.vmap(lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1]))
        #prec = -hess_fn(hess) # (n_sigs, n_arms, n_arms)

        #order = jnp.arange(n_arms)
        #q1 = jnp.where(order == 0, -1, 0)
        #q1 = jnp.where(jnp.arange(n_arms) == arm, 1, q1)

        #loc = x_max @ q1
        #scale = jnp.linalg.solve(prec, q1[None,:]) @ q1
        #normal_term = jax.scipy.stats.norm.cdf(thresh, loc=loc, scale=scale)
        #out = normal_term @ post_weighted

        #return out
        return 0

    def stage_1(self, p, key):
        """
        Runs a single simulation of Stage 1 of the Lei example.

        Parameters:
        -----------
        p:      simulation grid-point.
        key:    jax PRNG key.

        Returns:
        --------
        data, n_non_futile, non_futile_idx, pr_best, key

        data:           (number of arms, 2) where column 0 is the simulated binomial data for each arm
                        and column 1 is the corresponding value for the Binomial n parameter.
        n_non_futile:   number of non-futile treatment arms.
        non_futile_idx: vector of booleans indicating whether each arm is non-futile.
        pr_best:        vector containing probability of being the best arm for each arm.
                        It is set to jnp.nan if the arm was dropped for futility or if the arm is control (index 0).
        key:            last PRNG key used.
        """

        n_arms = self.n_arms
        n_stage_1 = self.n_stage_1
        n_interims = self.n_interims
        n_add_per_interim = self.n_add_per_interim
        futility_threshold = self.futility_threshold

        # create initial data
        n_arr = jnp.full(shape=n_arms, fill_value=n_stage_1)
        data = dist.Binomial(total_count=n_arr, probs=p).sample(key)
        data = jnp.stack((data, n_arr))

        # auxiliary variables
        stage_1_not_done = True
        non_futile_idx = jnp.ones(n_arms, dtype=bool)
        pr_best = self.compute_pr_best(data, non_futile_idx)
        order = jnp.arange(0, len(non_futile_idx))

        # Stage 1:
        for _ in range(n_interims):
            # get non-futile arm indices (offset by 1 because of control arm)
            # force control arm to be non-futile
            non_futile_idx = jnp.where(order == 0, True, pr_best >= futility_threshold)

            # if no non-futile treatment arms, terminate trial
            # else if exactly 1 non-futile arm, terminate stage 1 by choosing that arm.
            n_non_futile = jnp.sum(non_futile_idx[1:])
            stage_1_not_done = n_non_futile > 1

            continue_idx = non_futile_idx & stage_1_not_done

            # evenly distribute the next patients across non-futile arms
            remainder = n_add_per_interim % n_arms
            n_new = jnp.where(
                continue_idx, n_add_per_interim // n_non_futile + (order < remainder), 0
            )
            _, key = jax.random.split(key)
            y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)
            data = data + jnp.stack((y_new, n_new), axis=-1)

            # compute probability of best for each arm
            pr_best = self.compute_pr_best(data, continue_idx)

        return data, n_non_futile, non_futile_idx, pr_best, key

    def stage_2(
        self,
        data,
        non_futile_idx,
        pr_best,
        p,
        key,
    ):
        """
        Runs a single simulation of stage 2 of the Lei example.

        Parameters:
        -----------
        data:   simulated binomial data as in lei_stage_1 output.
        non_futile_idx:         a boolean vector indicating which arm is non-futile.
        pr_best:                a vector of probability of each arm being best.
                                Assume to be only well-defined whenever non_futile_idx is True.
        p:                      simulation grid-point.
        key:                    jax PRNG key.

        Returns:
        --------
        0 if no rejection, otherwise 1.
        """
        n_stage_2 = self.n_stage_2
        pps_threshold_lower = self.pps_threshold_lower
        pps_threshold_upper = self.pps_threshold_upper
        posterior_difference_threshold = self.posterior_difference_threshold
        rejection_threshold = self.rejection_threshold

        # select best treatment arm based on probability of each arm being best
        # since non_futile_idx always treats control arm (index 0) as non-futile,
        # we read past it.
        pr_best_subset = jnp.where(non_futile_idx[1:], pr_best[1:], 0)
        best_arm = jnp.argmax(pr_best_subset) + 1

        # add n_stage_2 number of patients to each of the control and selected treatment arms.
        n_new = jnp.where(non_futile_idx, n_stage_2, 0)
        _, key = jax.random.split(key)
        y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)

        # pool outcomes for each arm
        data = data + jnp.stack((y_new, n_new), axis=-1)

        pps = self.compute_pps(data)

        # check early-stop based on futility (lower) or efficacy (upper)
        early_stop = (pps < pps_threshold_lower) | (pps > pps_threshold_upper)

        return jax.lax.cond(
            early_stop,
            lambda: False,
            lambda: (
                self.posterior_difference(data, best_arm, posterior_difference_threshold)
                < rejection_threshold
            ),
        )

    def single_sim(self, p, key):
        """
        Runs a single simulation of both stage 1 and stage 2.

        Parameters:
        -----------
        p:      simulation grid-point.
        key:    jax PRNG key.
        """

        # Stage 1:
        data, n_non_futile, non_futile_idx, pr_best, key = self.stage_1(
            p=p,
            key=key,
        )

        # Stage 2 only if no early termination based on futility
        return jax.lax.cond(
            n_non_futile == 0,
            lambda: False,
            lambda: self.stage_2(
                data=data,
                non_futile_idx=non_futile_idx,
                pr_best=pr_best,
                p=p,
                key=key,
            ),
        )

    def simulate_point(self, p, keys):
        single_sim_vmapped = jax.vmap(self.single_sim, in_axes=(None, 0))
        return single_sim_vmapped(p, keys)

    def simulate(self, grid_points, keys):
        simulate_point_vmapped = jax.vmap(self.simulate_point, in_axes=(0, None))
        return simulate_point_vmapped(grid_points, keys)

```

```python
%%time
params = {
    "n_arms" : 2,
    "n_stage_1" : 50,
    "n_interims" : 3,
    "n_add_per_interim" : 100,
    "futility_threshold" : 0.1,
    "n_stage_2" : 100,
    "pps_threshold_lower" : 0.1,
    "pps_threshold_upper" : 0.9,
    "posterior_difference_threshold" : 0.1,
    "rejection_threshold" : 0.05,
}

lei_obj = Lewis45(**params)
```

```python
%%time
p = jnp.zeros(2)
grid_points = jnp.array([p] * 1000)
n_sims = 1
keys = jax.random.split(key, num=n_sims)
```

```python
%%time
#rejections = jax.jit(lei_obj.single_sim)(p, key)
rejections = jax.jit(lei_obj.simulate_point)(p, keys)
#rejections = jax.jit(lei_obj.simulate, static_argnums=(0, 3))(n_sims, grid_points, key, 1)
```

```python
#jnp.sum(rejections)
rejections
```
