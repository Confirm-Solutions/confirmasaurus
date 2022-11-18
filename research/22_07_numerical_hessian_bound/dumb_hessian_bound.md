## A dumb hessian bound


\begin{align}
\frac{d^2f(\theta)}{d\theta^2} &= \int_{x \in F(\theta)} \frac{d^2 p}{d\theta^2} (x | \theta) dx \tag*{(exchange int/diff)}\\
\end{align}

\begin{align}
\frac{d^2f(\theta)}{d\theta^2} &\leq \int_{x \in F(\theta)} \max(0, \frac{d^2 p}{d\theta^2} (x | \theta)) dx  \tag*{(drop negative values)}\\ 
\end{align}

\begin{align}
\frac{d^2f(\theta)}{d\theta^2} &\leq \int_{x \in \Omega} \max(0, \frac{d^2 p}{d\theta^2} (x | \theta)) dx \tag*{(expand integration domain)}\\
\end{align}


Ending this one day mini-project since I got where I wanted to be:

I think we can probably do 2nd order bounds for any distribution with two well behaved derivatives.
the approach below requires doing a bunch of numerical integration, but the computational expense should be tiny compared to the cost of the 0/1 order bounds that require simulation.
but I think it’s nice to be able to honestly advertise that we can handle a much broader set of potential outcomes, with the qualification that we’d have to build some tools to do it reliably on a real problem.

Early this morning, I played around with doing a fully numerical hessian bound for the binomial case. Obviously this isn’t useful for binomial since we have a better bound, but it’s a case where I can compare to what we have already. 

Anyway, there are sort of three steps:
1. we can directly calculate the integral over the possible data!
2. then, we can max(0, hessian) to get an upper bound.
3. then, we can convert to a “fancy softmax” that gets us a smooth and differentiable upper bound. this is the first figure.

Then, the second figure shows:
1. actual bound as a function of the logit-space parameter, not too much worse than the analytical bound.
1. the analytical binomial variance bound derived for all exponential family (here 0.5 * n * p * (1 - p))
1. the gradient and hessian *of the bound*, computed with jax…, these required computing third and fourth derivatives of the mass function. these would be useful if we wanted to find the maximum of the bound over a tile.


## Weibull

```sage
from sage.all import *
import matplotlib.pyplot as plt
import numpy as np


def show_figure(fig):

    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
```

```sage
x, k, beta, L = var("x, k, beta, L")
assume(k, "integer")
assume(k - 1 > 0)
# assume(k > 0)
assume(beta > 0)
beta = 1
```

```sage
p = k * beta * (x * beta) ** (k - 1) * exp(-((x * beta) ** k))
d2pdk2 = diff(diff(p, k), k)
```

```sage
p._sympy_()
```

```sage
d2pdk2._sympy_()
```

```sage
pos = p * (1 / k + log(x) * (1 - x**k)) ** 2
neg = p * (1 / (k**2) + x**k * log(x) ** 2)
```

```sage
# check that the pos, neg split matches the full expression
assert (d2pdk2 - (pos - neg)).full_simplify() == 0

# check the mean
mean = (p * x).subs(k == 0.5).nintegral(x, 0, 1000)
true_mean = 1.0 / beta * gamma(1 + 1 / 0.5)
print(mean[0], true_mean)

# look at pos, neg bound terms
def check(kv, a, b):
    P = pos.subs(k == kv).nintegral(x, a, b)
    N = neg.subs(k == kv).nintegral(x, a, b)
    return P[0], N[0]


for kv in [0.5, 1.0, 1.5]:
    print(check(kv, 0, 1000))
```

```sage
for kv in np.linspace(0.5, 2.5, 6):
    pos_bound = pos.subs(k == kv).nintegral(x, 0, 1000)
    max0_bound = max_symbolic(0, d2pdk2).subs(k == kv).nintegral(x, 0, 1000)
    print(
        f"k={kv:.2f}, positive part bound={pos_bound[0]:.2f}, max0 bound={max0_bound[0]:.2f}"
    )
```

```sage
# Why does max(0, ...) work well? Lots of integrand lies below zero!
myplot = plot((pos - neg).subs(k == 0.5), (x, 0, 5))
pm = myplot.matplotlib()
show_figure(pm)
```

## Binomial

```sage
n, k, p = var("n, k, p")
```

```sage
# pmf = binomial(n, k) * p ** k * (1 - p) ** (n - k)
pmf = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1)) * p**k * (1 - p) ** (n - k)
H = diff(diff(pmf, p), p)
boundH = max_symbolic(0, H)
```

```sage
eta = var("eta")
pmf_eta = pmf.subs(p == 1 / (1 + exp(-eta)))
```

```sage
Heta = diff(diff(pmf_eta, eta), eta)
boundHeta = max_symbolic(0, Heta)
```

```sage
tx = k
np = log(p / (1 - p))
hx = binomial(n, k)
Ap = -n * log(1 - p)
```

```sage
pmf2 = hx * exp(np * tx - Ap)
H2 = ((tx - diff(Ap, p)) ** 2 - diff(diff(Ap, p), p)) * pmf2
```

```sage
# plt.plot([H.subs(n == 100, p == 0.5, k == kv) for kv in range(0, 101)], 'r-')
# plt.plot([H2.subs(n == 100, p == 0.5, k == kv) for kv in range(0, 101)], 'r-')
# plt.plot([boundH.subs(n == 100, p == 0.5, k == kv) for kv in range(0, 101)], 'k-')
plt.plot(
    [
        boundHeta.subs(n == 100, eta == log(0.5 / (1 - 0.5)), k == kv)
        for kv in range(0, 101)
    ],
    "k-",
)
plt.show()
```

```sage
float(boundHeta.subs(n == 100, eta == log(0.5 / (1 - 0.5)), k == 40))
```

```sage
import numpy as np
```

```sage
nv = 100
for pv in np.linspace(0, 1, 20)[1:-1]:
    bound = np.sum(
        [
            float(boundHeta.subs(n == nv, eta == log(pv / (1 - pv)), k == kv))
            for kv in range(0, nv + 1)
        ]
    )
    simple_bound = 0.5 * nv * pv * (1 - pv)
    print(bound, simple_bound)
```

## Is it possible to numerically integrate a PMF like the binomial one?

yes, sort of.

```sage
nv = 100
pv = 0.5
full_entries = [
    float(boundHeta.subs(n == nv, eta == log(pv / (1 - pv)), k == kv))
    for kv in range(0, nv + 1)
]
full = np.sum(full_entries)
full
```

```sage
np.arange(0, nv + 1)[np.where(np.array(full_entries) == 0)]
```

```sage
ogx, ogw = np.polynomial.legendre.leggauss(10)


def to_interval(a, b):
    return (ogx + 1) / 2 * (b - a) + a, ogw / 2 * (b - a)


def integrate(f, gx, gw):
    return np.sum(gw * [f.subs(k == kv) for kv in gx])


f = boundHeta.subs(n == nv, eta == log(pv / (1 - pv)))
v1 = integrate(f, *to_interval(0, 45)) + integrate(f, *to_interval(55, 100))

v2 = f.nintegral(k, 0, 100)

v1, v2[0]
```

Note that this numerical integral above *has* converged. The reason it disagrees with the explicit pmf computation is that the domain is slightly different: integers vs reals.

Is this useful? Dunno, maybe.

```sage
show_figure(plot(f, (k, 0, 100)).matplotlib())
```

## JAX stuff

```sage
import jax
import numpyro.distributions as dist
import jax.numpy as jnp
```

```sage
def pmf_jax_probs(eta, n, k):
    p = jax.scipy.special.expit(eta)
    return jnp.exp(dist.Binomial(n, probs=p).log_prob(k))


# def pmf_jax_logits(eta, n, k):
#     return jnp.exp(dist.Binomial(n, logits=eta).log_prob(k))
# dist.Binomial(100, probs=0.5).log_prob(30), dist.Binomial(100, logits=0).log_prob(30)
# pmf2 = pmf_jax_logits(jax.scipy.special.logit(0.5), 100, np.arange(0, 101))
# plt.plot(pmf2, 'b-')
# H = jax.hessian(pmf_jax_probs)(jax.scipy.special.logit(p), 100, np.arange(0, 101))
# H2 = jax.hessian(pmf_jax_logits)(jax.scipy.special.logit(p), 100, np.arange(0, 101))
# plt.plot(H)
# plt.plot(H2)
# plt.show()

p = 0.3
pmf = pmf_jax_probs(jax.scipy.special.logit(p), 100, np.arange(0, 101))
plt.plot(pmf, "k-")
plt.show()
```

```sage
H = jax.hessian(pmf_jax_probs)(jax.scipy.special.logit(p), 100, np.arange(0, 101))
plt.plot(H)
plt.show()
```

```sage
plt.plot(jnp.maximum(H, 0))
plt.show()
```

```sage
xs = np.linspace(-4, 4, 100)
plt.plot(xs, jax.nn.softplus(xs))
plt.plot(xs, jax.nn.relu(xs))
plt.show()
```

```sage
plt.plot(jax.nn.softplus(H), label="fancy softplus")
plt.plot(jnp.maximum(H, 0), label="max0")
plt.legend()
plt.show()
```

```sage
plt.plot(jax.nn.softplus(H * 3) / 3, label="fancy softplus")
plt.plot(jnp.maximum(H, 0), label="max0")
plt.legend()
plt.show()
```

```sage
plt.plot(jax.nn.softplus(0.1 * H / pmf) * pmf / 0.1, label="fancy softplus")
plt.plot(jnp.maximum(H, 0), label="max0")
plt.legend()
plt.show()
```

```sage
H = jax.hessian(pmf_jax_probs)(jax.scipy.special.logit(p), 100, np.arange(0, 101))
pmf = pmf_jax_probs(jax.scipy.special.logit(p), 100, np.arange(0, 101))
beta = np.max(pmf)
plt.plot(jax.nn.softplus(beta * H / pmf) * pmf / beta, label="fancy softplus")
plt.plot(jnp.maximum(H, 0), label="max0")
plt.plot(H, label="true hessian")
plt.legend()
plt.ylim([-0.5, 1.0])
plt.xlabel("$y$")
plt.ylabel("$p(y|n=100, p=0.3)$")
plt.show()
```

```sage
# but max(pmf) is not smooth, so fit a polynomial
def max_pmf(eta):
    return jnp.max(pmf_jax_probs(eta, 100, np.arange(0, 101)))


etas = np.linspace(-10, 10, 200)
max_pmf = jax.vmap(max_pmf)(etas)
max_poly = np.polyfit(etas, max_pmf, 11)
plt.plot(etas, max_pmf)
plt.plot(etas, jnp.polyval(max_poly, etas) * 2)
plt.show()
```

```sage
p = 0.04
pmf = pmf_jax_probs(jax.scipy.special.logit(p), 100, np.arange(0, 101))
H = jax.hessian(pmf_jax_probs)(jax.scipy.special.logit(p), 100, np.arange(0, 101))
beta = 2 * np.max(pmf)
plt.plot(jnp.maximum(H, 0))
plt.plot(jax.nn.softplus(beta * H / pmf) * pmf / beta)
plt.show()
```

```sage
pmf_jax = pmf_jax_probs
```

```sage
def bound_max0(eta, n, k):
    H = jax.hessian(pmf_jax)(eta, 100, np.arange(0, 101))
    max0H = jax.nn.relu(H)
    return jnp.sum(max0H)


bound_max0(jax.scipy.special.logit(0.5), 100, np.arange(0, 101))
```

```sage
def bound_softplus(eta, n, k):
    H = jax.hessian(pmf_jax)(eta, 100, np.arange(0, 101))
    eps = 1e-10
    pmf = pmf_jax(eta, 100, np.arange(0, 101))
    pmf = jnp.maximum(pmf, eps)
    beta = jnp.polyval(max_poly, eta) * 1.5
    softH = jax.nn.softplus(beta * H / pmf) * pmf / beta
    # softH = jnp.where(jnp.isnan(softH), 0, softH)
    return jnp.sum(softH)


bound_softplus(jax.scipy.special.logit(0.5), 100, np.arange(0, 101))
```

```sage
bound_f = bound_softplus
ps = np.linspace(0, 1, 200)[1:-1]
etas = jax.scipy.special.logit(ps)
bs = jax.vmap(bound_f, in_axes=(0, None, None))(etas, 100, np.arange(0, 101))
gs = jax.vmap(jax.grad(bound_f), in_axes=(0, None, None))(etas, 100, np.arange(0, 101))
hs = jax.vmap(jax.grad(jax.grad(bound_f)), in_axes=(0, None, None))(
    etas, 100, np.arange(0, 101)
)
```

```sage
import pandas as pd

df = pd.DataFrame(dict(p=ps, eta=etas, bound=bs, grad=gs, hess=hs))
df
```

```sage
plt.figure(figsize=(8, 8))
plt.title("n=100 binomial bounds")
plt.plot(df["eta"], 0.5 * 100 * df["p"] * (1 - df["p"]), label="cov bound")
plt.plot(df["eta"], df["bound"], label="softplus bound")
plt.plot(df["eta"], df["grad"], label="grad")
plt.plot(df["eta"], df["hess"], label="hessian")
plt.legend()
plt.xlabel("$\eta$")
plt.show()
```

```sage

```
