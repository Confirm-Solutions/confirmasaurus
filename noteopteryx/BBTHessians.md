**Facts about bayesian basket trial hessian matrices:**

For a while, in our berry/thall work, we’ve been assuming we have hessians that look like A + diag(b) and we know inv(A). This is a nice form for using Sherman-Morrison to invert since that gives efficient inverse for matrices like ($A + uv^T$) if we know $A^{-1}$. Unfortunately, as we've talked about a bunch, this method of inversion has poor numerical properties when $\sigma^2$ is small because $A^{-1}_{ij} = \sigma_\mu^2 + \sigma^2\delta_{ij}$ is just fundamentally problematic in 32 bit when we're trying to add $\sigma^2=1e-6$ to $\sigma_{\mu}^2=100$ with only 7 digits of precision. 

It turns out there's a better way. We still use Sherman Morrison, but actually formulate the problem a bit differently. If we go back and look at the relevant hessian matrices, they are of the form $A_{ij} = b + a_i\delta_{ij}$ That is, the entries are constant off the diagonal! So, we can use Sherman Morrison on the matrix $aI + sqrt(b)sqrt(b)^T$ You can see how this maps to the standard Sherman Morrison form above. This alternative formulation has a couple nice properties:
1. It avoids dealing with matrices that have involve adding small numbers to large numbers as was causing problems. Basically, the small and large numbers are explicitly kept separate!
2. It's really fast. Since the matrix term is diagonal, the inverse is very cheap. And, the rank 1 update term is a constant everywhere such that the sqrt(...) actually drops out. The result is something that's like 10 floating operations!

The end result is something that "works" even for weird cases. In most normal cases (sigma2 not tiny, y != 0), it works very well. In bad edge cases where y=0 and sig2=1e-6, the optimization convergence has like 1% error. Which is pretty bad for a numerical method, but actually probably acceptable in light of our plans for approximation error tests. In these bad cases, we fall back to a more accurate method. In particular, that fall back method can be 64 bit arithmetic. 

Despite being kind of "in the weeds", I think stuff like this is going to be useful as building blocks. Basket trials are a common form and will often have this matrix structure. 

Tons of credit to Alex for getting this 32 bit stuff 90% done!!