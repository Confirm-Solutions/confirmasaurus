```python
from imprint.nb_util import setup_nb
setup_nb()

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import imprint as ip
```

```python

@jax.jit
def _sim(outcome_samples, assignment, arm_counts, theta, null_truth):
    p = jax.scipy.special.expit(theta)
    assigned_p = jnp.take(p, assignment, axis=1)
    S = outcome_samples[None] < assigned_p
    successes = jnp.stack(
        [jnp.sum(S * (assignment[None] == i), axis=-1) for i in range(2)], axis=-1
    )
    pihat = successes / arm_counts[None, :, :]
    phat = jnp.sum(successes, axis=-1) / jnp.sum(arm_counts, axis=-1)
    Zpooled = (pihat[..., 1] - pihat[..., 0]) / jnp.sqrt(
        phat * (1 - phat) * (1 / arm_counts[..., 0] + 1 / arm_counts[..., 1])
    )
    eps = 1e-10
    Zpooled = jnp.where(
        (phat > eps) & (phat < (1 - eps)) & null_truth[:, 0, None], Zpooled, jnp.inf
    )
    pvalue = 1 - jax.scipy.stats.norm.cdf(Zpooled)
    return pvalue, Zpooled, successes


class Efron:
    def __init__(self, seed, max_K, *, n_patients):
        self.seed = seed
        self.max_K = max_K
        self.family = "binomial"
        self.family_params = {"n": n_patients}
        self.dtype = jnp.float64

        key = jax.random.PRNGKey(self.seed)
        key1, key2 = jax.random.split(key)
        self.outcome_samples = jax.random.uniform(
            key1, shape=(max_K, n_patients), dtype=self.dtype
        )
        self.assignment_samples = jax.random.uniform(
            key2, shape=(max_K, n_patients), dtype=self.dtype
        )
        self.arm_counts = np.zeros((max_K, 2))
        self.assignment = np.empty((max_K, n_patients), dtype=np.int32)
        for i in range(n_patients):
            arm0_prob = np.where(
                self.arm_counts[:, 0] == self.arm_counts[:, 1],
                0.5,
                np.where(
                    self.arm_counts[:, 0] < self.arm_counts[:, 1], 1.0 / 3.0, 2.0 / 3.0
                ),
            )
            self.assignment[:, i] = self.assignment_samples[:, i] < arm0_prob
            self.arm_counts[np.arange(max_K), self.assignment[:, i]] += 1
        self.assignment = jnp.array(self.assignment)

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        return _sim(*self.args(begin_sim, end_sim, theta, null_truth))[0]
            
    def args(self, begin_sim, end_sim, theta, null_truth):
        return (
            self.outcome_samples[begin_sim:end_sim],
            self.assignment[begin_sim:end_sim],
            self.arm_counts[begin_sim:end_sim],
            theta,
            null_truth
        )

e = Efron(0, 10, n_patients=50)
theta = jnp.array([[0.0, 0.0]])
null_truth = jnp.array([[True]])
pvalue, Z, successes = _sim(*e.args(0, e.max_K, theta, null_truth))
diff = successes[0,..., 0] - successes[0,..., 1]

plt.plot(diff, Z[0], 'ko')
plt.xlabel("Difference in successes")
plt.ylabel("Pooled negative Z-stat")
plt.show()
plt.plot(diff, pvalue[0], 'ko')
plt.xlabel("Difference in successes")
plt.ylabel("p-value")
plt.show()
```

```python
g = ip.cartesian_grid([-1, -1], [1, 1], n=[800, 800], null_hypos=[ip.hypo("theta0 > theta1")])
rej_df = ip.validate(Efron, g=g, lam=0.025, K=2**17, model_kwargs=dict(n_patients=40))
```

```python
rej_df['tie_est'].max(), rej_df['tie_bound'].max()
```

```python
s = 3
plt.figure(figsize=(10, 10), constrained_layout=True)
plt.subplot(2,2,1)
plt.title('TIE Estimate')
plt.scatter(g.df['theta0'], g.df['theta1'], c=rej_df['tie_est'], s=s)
plt.colorbar()
plt.ylabel('$\\theta_1$')
plt.subplot(2,2,2)
plt.title('CSE bound')
plt.scatter(g.df['theta0'], g.df['theta1'], c=rej_df['tie_bound'], s=s)
plt.colorbar()
plt.subplot(2,2,3)
plt.title('Clopper-Pearson bound cost')
plt.scatter(g.df['theta0'], g.df['theta1'], c=rej_df['tie_cp_bound'] - rej_df['tie_est'], s=s)
plt.colorbar()
plt.xlabel('$\\theta_0$')
plt.ylabel('$\\theta_1$')
plt.subplot(2,2,4)
plt.title('CSE bound cost')
plt.scatter(g.df['theta0'], g.df['theta1'], c=rej_df['tie_bound'] - rej_df['tie_cp_bound'], s=s)
plt.colorbar()
plt.xlabel('$\\theta_0$')
plt.show()
```

```python
lams_df = ip.calibrate(Efron, g=g, alpha=0.025, model_kwargs=dict(n_patients=40))
```

```python
plt.scatter(g.df['theta0'], g.df['theta1'], c=lams_df['lams'])
plt.colorbar()
plt.show()
```

```python
lams_df['lams'].min()
```

## Debugging version

```python
n_patients = 50

# assignment RVs
phi = np.random.uniform(0, 1, n_patients)
arm_counts = [0, 0]
assignment = np.empty(n_patients, dtype=np.int32)
for i in range(n_patients):
    if arm_counts[0] == arm_counts[1]:
        # 50/50 chance of arm 0 and arm 1
        thresh = 0.5
    elif arm_counts[0] < arm_counts[1]:
        # 2/3rd chance of arm 0 and 1/3rds chance of arm 1
        thresh = 1.0 / 3.0
    else:
        # 1/3rd chance of arm 0 and 2/3rds chance of arm 1
        thresh = 2.0 / 3.0

    assignment[i] = int(phi[i] < thresh)
    arm_counts[assignment[i]] += 1
```

```python
np.random.seed(0)
outcomes = np.random.uniform(0, 1, n_patients)
p = np.array([0.2, 0.4])
S = outcomes < p[assignment]
successes = np.array([np.sum(S * (assignment == i)) for i in range(2)])
successes
```

```python
import imprint as ip
g = ip.cartesian_grid([-1], [1], n=[100], null_hypos=[ip.hypo("x < 0")])
tune_df = ip.calibrate(Efron, g, model_kwargs=dict(n=50))
lam = tune_df["lams"].min()
print(lam)
```
