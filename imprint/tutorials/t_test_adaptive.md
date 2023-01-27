# $t$ Test Adaptive

```python
from imprint.nb_util import setup_nb

# setup_nb is a handy function for setting up some nice plotting defaults.
setup_nb()
import scipy.stats
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

import imprint as ip
from imprint.models.ttest_adaptive import TTest1DAda
```

In this tutorial, we study a generalization of the $t$ test procedure with interim analyses.
Let $i=1,\ldots, I$ denote the interims where $I \geq 0$.
For each interim $1 \leq i \leq I$, 
we perform an analysis by combining the data observed so far and running the usual $t$ test.
If the procedure does not reject, we add $n_i$ additional samples and continue to the next interim (or final analysis).
Otherwise, we finish the entire procedure.
If the last interim analysis $I$ does not reject and $n_I$ samples are added, 
we move onto the final analysis, which again runs the usual $t$ test.

Let $N_i = \sum\limits_{j=0}^{i} n_j$ for $i=0,\ldots, I$, 
where $n_0$ is the initial number of samples and $n_i$ are the additional samples after interim analysis $i$.
$N_i$ is then the total number of samples seen after interim analysis $i$.
For each $i = 0,\ldots, I$,
we sample $n_i$ draws $X_{N_{i-1} + 1}, \ldots, X_{N_{i}} \sim \mathcal{N}(\mu, \sigma^2)$.
We show a recursive formula for the $T$ statistic at each interim:
let $T^i$ denote the test statistic at stage $i$ ($0 \leq i \leq I$) with $N_i$ samples.
Then,
$$
\begin{align*}
    T^i &:= \frac{A_i - N_i(\mu - \mu_0)}{N_i \sqrt{\frac{B_i}{N_i-1}}} \\
    A_i &:= \sum\limits_{j=1}^{N_i} X_j \\
    B_i &:= \sum\limits_{j=1}^{N_i} (X_j - \bar{X}_i)^2 \\
    \bar{X}_i &:= \frac{A_i}{N_i}
\end{align*}
$$

First, we clearly have
$$
\begin{align*}
    A_i &:= A_{i-1} + \Delta_i \\
    \Delta_i &:= \sum\limits_{j=N_{i-1}+1}^{N_i} X_j \sim \mathcal{N}(n_i \mu, n_i \sigma^2) \perp\!\!\!\perp A_{i-1}
\end{align*}
$$

Next, we have that
$$
\begin{align*}
    B_i 
    &:= 
    \sum\limits_{j=1}^{N_i} (X_j - \bar{X}_i)^2 
    =
    \sum\limits_{j=1}^{N_{i-1}} (X_j - \bar{X}_i)^2
    +
    \sum\limits_{j=N_{i-1}+1}^{N_{i}} (X_j - \bar{X}_i)^2
    \\&=
    \sum\limits_{j=1}^{N_{i-1}} (X_j - \bar{X}_{i-1})^2
    + N_{i-1} (\bar{X}_{i-1} - \bar{X}_i)^2
    + \sum\limits_{j=N_{i-1}+1}^{N_{i}} \left(X_j - \frac{\Delta_i}{n_i}\right)^2
    + n_i \left(\frac{\Delta_i}{n_i} - \bar{X}_i\right)^2
\end{align*}
$$
Note that
$$
\begin{align*}
    \bar{X}_i - \bar{X}_{i-1}
    &=
    \left(\frac{1}{N_i} - \frac{1}{N_{i-1}}\right) A_{N_i}
    + \frac{1}{N_{i-1}} \Delta_i
    =
    \frac{n_i}{N_{i-1}} \left(\frac{\Delta_i}{n_i} - \bar{X}_i\right)
\end{align*}
$$
So,
$$
\begin{align*}
    B_i 
    &=
    B_{i-1} + S_i + \frac{n_i N_i}{N_{i-1}} \left(\frac{\Delta_i}{n_i} - \bar{X}_i\right)^2
    =
    B_{i-1} + S_i + \frac{n_i N_{i-1}}{N_i} \left(\frac{\Delta_i}{n_i} - \frac{A_{i-1}}{N_{i-1}}\right)^2
    \\
    S_i 
    &:= 
    \sum\limits_{j=N_{i-1}+1}^{N_{i}} \left(X_j - \frac{\Delta_i}{n_i}\right)^2
    \sim \sigma^2 \chi^2_{n_i-1} \perp\!\!\!\perp \Delta_i
\end{align*}
$$

```python
mu_0 = 0  # fixed threshold for null hypothesis
theta_min = [-1, -1]  # minimum for theta
theta_max = [0, -0.1]  # maximum for theta
n_init = 20  # initial number of Gaussian draws
n_samples_per_interim = 20  # number of Gaussian draws per interim
n_interims = 1  # number of interims
n_gridpts = [100, 100]  # number of grid-points along each direction
alpha = 0.025  # target nominal level
n_sims = 8192  # number of simulations

# try true critical threshold when 0 interims in the lambda space
lam = -scipy.stats.t.isf(alpha, df=n_samples - 1)
```

```python
grid = ip.cartesian_grid(
    theta_min=theta_min,
    theta_max=theta_max,
    n=n_gridpts,
    null_hypos=[ip.hypo(f"theta0 <= {-2 * mu_0} * theta1")],
)
```

```python
rej_df = ip.validate(
    TTest1DAda,
    grid,
    lam,
    K=n_sims,
    model_kwargs={
        "n_init": n_init,
        "n_samples_per_interim": n_samples_per_interim,
        "n_interims": n_interims,
        "mu0": mu_0,
    },
)
rej_df.tail()
```
