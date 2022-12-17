```R
library(INLA)
```

```R
n_arms = 4
y = c(0, 1, 9, 10)
n = c(20, 20, 35, 35)
# y = c(11, 11, 10, 9)
# n = c(35, 35, 35, 35)
df <- data.frame(y = y, gid = (1:n_arms), const=rep(1, n_arms))

mu_0 = -1.34
mu_sig2 = 100.0
sig2_alpha = 0.0005
sig2_beta = 0.000005
logit_p1 = rep(qlogis(0.3), n_arms)

```

```R
# Run INLA with a model where logit(p) = intercept + random_effect(drawn iid from normal)
# The prior on the random effect influences the sharing between groups. I
# haven't tuned this yet.

# theta = offset(mu_0 + logit_p1) + N(mu=0, sig2=S^2*J + sig2*I)
# sig2 ~ invgamma(alpha, beta)
# J = all ones
# I = diagonal
df <- data.frame(y = y, gid = (1:n_arms), const=rep(1, n_arms))
expr = y ~ 0 + f(
    gid, 
    model = "iid",
    hyper = list(prec = list(param = c(sig2_alpha, sig2_beta)))
) + f(
    const,
    model="iid",
    hyper = list(prec = list(initial=1.0 / mu_sig2, fixed=TRUE))
)
result <- inla(
    expr,
    data=df,
    Ntrials = n,
    family = "binomial",
    offset=mu_0 + logit_p1,
    control.compute = list(
        config=TRUE,
        return.marginals=TRUE,
        return.marginals.predictor=TRUE
    ),
    control.inla = list(),
    control.predictor = list(
        precision=1e9
    )
)
summary(result)
hyperpar_data = result$marginals.hyperpar
write.csv(hyperpar_data[[1]], "hyperpar_data2.csv")
```

```R
result$joint.hyper
```

```R
exp(-result$mode$theta)
```

```R
result$mode$x
```

```R
result$mode
```

```R
mu_0 + logit_p1
```

```R
result$mode$x[5:8] + result$mode$x[9] + mu_0 + logit_p1
```

```R
configs = result$misc$configs$config
names(result$misc$configs)
```

```R
length(configs)
```

```R
length(configs[[10]]$mean)
```

```R
sig2 = exp(-configs[[10]]$theta[[1]])
```

```R
diag(sig2)
```

```R
cov = matrix(mu_sig2, nrow=4, ncol=4) + diag(sig2)
cov
```

```R
configs[[10]]$Q
```

```R
list(1.0 / exp(configs[[1]]$theta[[1]]), configs[[1]]$mean[1:4], configs[[1]]$mean[5:8] + configs[[1]]$mean[9])
```

```R
# logprec = sapply(configs, function(x) x$theta[[1]])
# logpost = sapply(configs, function(x) x$log.posterior[[1]])
# write.csv(data.frame(logprec = logprec, logpost = logpost), "hyperpar_data.csv")
```

```R
sig2_sample = 1.0 / inla.hyperpar.sample(n=100000, result)
hist(log10(sig2_sample), breaks = 100)
hyperpar_data = result$marginals.hyperpar
hyperprec = hyperpar_data[[1]][, 'x']
hypersig2 = 1.0 / hyperprec
pdf = hyperpar_data[[1]][, 'y'] / hypersig2^2
plot(log10(hypersig2), log10(pdf), type = "l")
```

## Old stuff

```R
# result.samp <- inla.posterior.sample(100, result)
# names(result.samp[[1]])
# print(result.samp[[1]])

```

```R
# Plot marginal PDFs.
# These will depend heavily on hyperparameter priors which I have just left
# default for now.
par(mfrow=c(2,2))
for (i in 1:n_groups) {
    theta_i <- result$marginals.linear.predictor[[i]][,1]
    density <- result$marginals.linear.predictor[[i]][,2]
    plot(theta_i, density, main=t_i[[i]])
    abline(v=mean(t_i), col="blue")
}
```

```R
# Print 95% confidence intervals for the linear predictors. 
# I'm not 100% sure these are correct, but they track correctly with the y_i
for (i in 1:n_groups) {
    print(t_i[[i]])
    print(inla.hpdmarginal(0.95, result$marginals.linear.predictor[[i]]))
}
```

```R
for (i in 1:n_groups) {
    print(y_i[[i]])
    print(inla.pmarginal(0, result$marginals.linear.predictor[[i]]))
}
```

```R
# expr2 = y ~ 0 + offset(mu_0 + logit_p1) + f(
#     gid, 
#     model="generic3", 
#     Cmatrix= list(Diagonal(4, 1.0), matrix(data=1.0 / 100.0, nrow=4, ncol=4)), 
#     hyper = list(
#         theta1=list(param = c(sig2_alpha, sig2_beta), fixed=FALSE),
#         theta2=list(fixed=TRUE, initial=1.0),
#         theta11=list(fixed=TRUE, initial=1.0)
#     )
# )
```

```R
result <- inla(
    y ~ 0,
    data=data.frame(y=c(2,3,4)),
    Ntrials=rep(10, 3),
    family="binomial",
    control.compute = list(
        config=TRUE
    )
)
result$mode
```

```R
result <- inla(
    y ~ 0 + offset(rep(1,4)),
    data=data.frame(y=c(1,2,3,4)),
    Ntrials=rep(10, 4),
    family="binomial",
    control.compute = list(
        config=TRUE,
        return.marginals=TRUE,
        return.marginals.predictor=TRUE,
        q=TRUE
    ),
    verbose=TRUE,
    num.threads=1
)
names(result)
```
