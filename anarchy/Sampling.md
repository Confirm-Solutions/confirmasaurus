Mike and I were talking earlier in the week about how to sample from an INLA posterior. I read through the R-inla code to see how they do sampling. Here’s how they do the sampling:
- they sample from the list of hyperparameters according to the hyperparameter posterior
- then they sample the latent variables from the gaussian density at that value of the hyperparameter.
They don’t have an equivalent method that samples from a full INLA latent marginal.

This morning I realized how to do the INLA sampling correctly for the full laplace case. The procedure is very similar to above. Choose the latent variable marginal that you care most about being sampled correctly. Let’s say we choose theta_i
- sample one of the hyperparameter (sigma2 in berry) values
- sample one of the theta_i according to the grid that has already been determined by the hyperparam values during the latent marginal calculation
- then sample the remaining theta_{-i} according to the gaussian that was used in calculating the latent marginal.
No new distributions need to be calculated to do this since all these pieces are required to compute an accurate latent variable marginal.

I would expect that any INLA-style inference method can be transformed into a sampler using logic similar to this: just sample in the same order that the corresponding inference structures the conditional distributions.

Ref:
https://github.com/inbo/INLA/blob/0fa332471d2e19548cc0f63e36873e31dbd685be/R/posterior.sample.R#L193