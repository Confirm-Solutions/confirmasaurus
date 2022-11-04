---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: R
    language: R
    name: ir
---

```R vscode={"languageId": "r"}
library(INLA)

```

```R vscode={"languageId": "r"}
n_arms = 4
y = c(0, 1, 9, 10)
n = c(20, 20, 35, 35)
df <- data.frame(y = y, gid = (1:n_arms), const=rep(1, n_arms))

mu_0 = -1.34
mu_sig2 = 100.0
sig2_alpha = 0.0005
sig2_beta = 0.000005
logit_p1 = rep(qlogis(0.3), n_arms)
```

```R vscode={"languageId": "r"}
# there's a great R-INLA vignette that explains how to use `inla.rgeneric.define(...)`!
# this is following the instructions there basically to the letter.
berry.model = function(
        cmd = c("graph", "Q", "mu", "initial", "log.norm.const", "log.prior", "quit"),
        theta = NULL)
{
    d = 4
    mu_sig2 = 100.0
    sig2_alpha = 0.0005
    sig2_beta = 0.000005
    envir = parent.env(environment())
    interpret.theta = function () {
        return (list(prec = exp(theta[1L])))
    }
    graph = function(){
        return (matrix(1, nrow=d, ncol=d))
    }
    Q = function() {
        p = interpret.theta()
        sig2 = 1.0 / p$prec
        cov = matrix(mu_sig2, nrow=d, ncol=d) + sig2 * Diagonal(d)
        q = solve(cov)
        return (inla.as.sparse(q))
    }
    mu = function() { return (numeric(0))  }
    log.norm.const = function() { return (numeric(0)) }
    log.prior = function() { 
        p = interpret.theta()$prec
        val = dgamma(p, shape=sig2_alpha, rate=sig2_beta, log=TRUE) + theta[1L]
        return (val)
    }
    initial = function() { 
        return (rep(1, 1)) 
    }
    quit = function() { return (invisible()) }

    if (!length(theta)) {
        theta = initial()
    }

    val = do.call(match.arg(cmd), args = list())

    return (val)
}
```

```R vscode={"languageId": "r"}

model = inla.rgeneric.define(berry.model)
use_custom = TRUE

# both these models should be producing the same output but there's noise
# that's added by the non-custom version because of the large values added to
# the diagonal of the precision matrix.
if (use_custom) {
    expr = y ~ 0 + f(gid, model=model)
    get_mode = function (x) {
        out = x[5:8] + mu_0 + logit_p1
        return (out)
    }
} else {
    expr = y ~ 0 + f(
        gid, 
        model = "iid",
        hyper = list(prec = list(param = c(sig2_alpha, sig2_beta)))
    ) + f(
        const,
        model="iid",
        hyper = list(prec = list(initial=1.0 / mu_sig2, fixed=TRUE))
    )
    get_mode = function (x) {
        out = x[5:8] + mu_0 + logit_p1 + x[9]
        return (out)
    }
}
result <- inla(
    expr,
    data=df,
    Ntrials = n,
    family = "binomial",
    offset = mu_0 + logit_p1,
    control.compute=list(
        config=TRUE,
        return.marginals=TRUE,
        return.marginals.predictor=TRUE
    ),
    control.inla=list(strategy="gaussian"),
    verbose=TRUE
)
summary(result)
```

```R vscode={"languageId": "r"}
c_sig2 = c()
mean = c()
for (i in 1:length(result$misc$configs$config)) {
    c_sig2[[i]] = exp(-result$misc$configs$config[[i]]["theta"][[1]][[1]])
    mean[[i]] = get_mode(result$misc$configs$config[[i]]["mean"][[1]])
}
list(sig2_mode = exp(-result$mode$theta[[1]]), latent_mode=get_mode(result$mode$x), sig2_grid=c_sig2)

hyperpar_data = result$marginals.hyperpar
write.csv(hyperpar_data[[1]], "hyperpar_data2.csv")
# Plot marginal PDFs.
# These will depend heavily on hyperparameter priors which I have just left
# default for now.
par(mfrow=c(2,2))
for (i in 1:4) {
    theta_i <- result$marginals.linear.predictor[[i]][,1]
    density <- result$marginals.linear.predictor[[i]][,2]
    plot(theta_i, density, main=theta_i[which.max(density)], xlab=i)
    # abline(v=mean(t_i), col="blue")
}
```

```R vscode={"languageId": "r"}
# grid_opts = c("ccd", "grid", "eb")
grid_opts = c("grid")
m.strategy <- lapply(c("gaussian", "simplified.laplace", "laplace"), 
  function(st) {
    return(lapply(grid_opts, function(int.st) {
      return (inla(
          expr,
          data=df,
          Ntrials = n,
          family = "binomial",
          offset = mu_0 + logit_p1,
          control.compute=list(
              config=TRUE,
              return.marginals=TRUE,
              return.marginals.predictor=TRUE
          ),
          control.inla = list(
            strategy = st,
            int.strategy = int.st,
            npoints=100,
            cutoff=1e-7
          )
        ))
    }))
})
```

```R vscode={"languageId": "r"}
# took this snippet from the source of https://becarioprecario.bitbucket.io/inla-gitbook/ch-INLA.html
library(ggplot2)
library("viridis")
intst.colors <- magma(4)

marg.theta <- lapply(m.strategy, function(X) {
  do.call(rbind, lapply(X, function(Y) {Y$marginals.random[[1]][[1]]}))
})
marg.theta <- as.data.frame(do.call(rbind, marg.theta))
ntheta1 = dim(m.strategy[[1]][[1]]$marginals.random[[1]][[1]])[[1]]
marg.theta$strategy <- rep(c("gaussian", "simplified.laplace", "laplace"),
  each = length(grid_opts) * ntheta1)
marg.theta$strategy <- factor(marg.theta$strategy,
  levels = c("gaussian", "simplified.laplace", "laplace"))
marg.theta$int.strategy <- rep(rep( grid_opts, each = ntheta1), 3)
marg.theta$int.strategy <- factor(marg.theta$int.strategy,
  levels = grid_opts)


ggplot(marg.theta, aes(x = x, y = y, linetype = strategy, colour = int.strategy)) +
    geom_point(size=1) + 
    geom_line(size=0.5) +
    xlim(-5.0, 3.0) + ylim(0, 0.4) +
    xlab(expression(theta[0])) +
    ylab(expression(paste(pi, "(", theta[0], " | ", bold(y), ")")) ) +
    scale_linetype_manual(values = c("solid", "dashed", "dotted")) +
    scale_colour_manual(values = intst.colors) +
    theme(legend.position = c(0.15, 0.65),
        legend.background = element_rect(fill = "transparent"),
        legend.title = element_blank())
```

```R vscode={"languageId": "r"}
write.csv(marg.theta, "latent_data.csv")
```
