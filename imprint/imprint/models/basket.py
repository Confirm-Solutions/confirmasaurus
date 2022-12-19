from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.stats
from scipy.special import logit


class BayesianBasket:
    def __init__(self, seed, K, *, n_arm_samples=35):
        self.n_arm_samples = n_arm_samples
        np.random.seed(seed)
        self.samples = np.random.uniform(size=(K, n_arm_samples, 3))
        self.fi = FastINLA(n_arms=3, critical_value=0.95)
        self.family = "binomial"
        self.family_params = {"n": n_arm_samples}

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        # 1. Calculate the binomial count data.
        # The sufficient statistic for binomial is just the number of uniform draws
        # above the threshold probability. But the `p_tiles` array has shape (n_tiles,
        # n_arms). So, we add empty dimensions to broadcast and then sum across
        # n_arm_samples to produce an output `y` array of shape: (n_tiles,
        # sim_size, n_arms)

        p = scipy.special.expit(theta)
        y = np.sum(self.samples[None, begin_sim:end_sim] < p[:, None, None], axis=2)

        # 2. Determine if we rejected each simulated sample.
        # rejection_fnc expects inputs of shape (n, n_arms) so we must flatten
        # our 3D arrays. We reshape exceedance afterwards to bring it back to 3D
        # (n_tiles, sim_size, n_arms)
        y_flat = y.reshape((-1, 3))
        n_flat = np.full_like(y_flat, self.n_arm_samples)
        data = np.stack((y_flat, n_flat), axis=-1)
        test_stat_per_arm = self.fi.test_inference(data).reshape(y.shape)
        return np.min(
            np.where(null_truth[:, None, :], test_stat_per_arm, np.inf), axis=-1
        )


@dataclass
class QuadRule:
    pts: np.ndarray
    wts: np.ndarray


def gauss_rule(n, a=-1, b=1):
    """
    Points and weights for a Gaussian quadrature with n points on the interval
    (a, b)
    """
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = (pts + 1) * (b - a) / 2 + a
    wts = wts * (b - a) / 2
    return QuadRule(pts, wts)


def log_gauss_rule(N, a, b):
    A = np.log(a)
    B = np.log(b)
    qr = gauss_rule(N, a=A, b=B)
    pts = np.exp(qr.pts)
    wts = np.exp(qr.pts) * qr.wts
    return QuadRule(pts, wts)


class FastINLA:
    def __init__(
        self,
        n_arms=4,
        mu_0=-1.34,
        mu_sig2=100.0,
        sigma2_n=15,
        sigma2_bounds=(1e-6, 1e3),
        sigma2_alpha=0.0005,
        sigma2_beta=0.000005,
        p1=0.3,
        critical_value=0.85,
        opt_tol=1e-3,
    ):
        self.n_arms = n_arms
        self.mu_0 = mu_0
        self.mu_sig2 = mu_sig2
        self.logit_p1 = logit(p1)

        # For numpy impl:
        self.sigma2_n = sigma2_n
        self.sigma2_rule = log_gauss_rule(self.sigma2_n, *sigma2_bounds)
        self.arms = np.arange(self.n_arms)
        self.cov = np.full((self.sigma2_n, self.n_arms, self.n_arms), self.mu_sig2)
        self.cov[:, self.arms, self.arms] += self.sigma2_rule.pts[:, None]
        self.neg_precQ = -np.linalg.inv(self.cov)
        self.logprecQdet = 0.5 * np.log(np.linalg.det(-self.neg_precQ))
        self.log_prior = scipy.stats.invgamma.logpdf(
            self.sigma2_rule.pts, sigma2_alpha, scale=sigma2_beta
        )
        self.opt_tol = opt_tol
        self.thresh_theta = np.full(self.n_arms, logit(0.1) - self.logit_p1)
        self.critical_value = critical_value

    def rejection_inference(self, data):
        _, exceedance, _, _ = self.inference(data)
        return exceedance > self.critical_value

    def test_inference(self, data):
        _, exceedance, _, _ = self.inference(data)
        return 1 - exceedance

    def inference(self, data, thresh_theta=None):
        """
        Bayesian inference of a basket trial given data with n_arms.

        Returns:
            sigma2_post: The posterior density for each value of the sigma2
                quadrature rule.
            exceedances: The probability of exceeding the threshold for each arm.
            theta_max: the mode of p(theta_i, y, sigma^2)
            theta_sigma: the std dev of a gaussian distribution centered at the
                mode of p(theta_i, y, sigma^2)
            hess_inv: the inverse hessian at the mode of p(theta_i, y, sigma^2)
        """
        if thresh_theta is None:
            thresh_theta = self.thresh_theta

        # TODO: warm start with DB theta ?
        # Step 1) Compute the mode of p(theta, y, sigma^2) holding y and sigma^2 fixed.
        # This is a simple Newton's method implementation.
        theta_max, hess_inv = self.optimize_mode(data)

        # Step 2) Calculate the joint distribution p(theta, y, sigma^2)
        logjoint = self.log_joint(data, theta_max)

        # Step 3) Calculate p(sigma^2 | y) = (
        #   p(theta_max, y, sigma^2)
        #   - log(det(-hessian(theta_max, y, sigma^2)))
        # )
        # The last step in the optimization  will be sufficiently small that we
        # shouldn't need to update the hessian that was calculated during the
        # optimization.
        # hess = np.tile(-precQ, (N, 1, 1, 1))
        # hess[:, :, arms, arms] -= (n[:, None] * np.exp(theta_adj) /
        # ((np.exp(theta_adj) + 1) ** 2))
        log_sigma2_post = logjoint + 0.5 * np.log(np.linalg.det(-hess_inv))
        # This can be helpful for avoiding overflow.
        # log_sigma2_post -= np.max(log_sigma2_post, axis=-1)[:, None] - 600
        sigma2_post = np.exp(log_sigma2_post)
        sigma2_post /= np.sum(sigma2_post * self.sigma2_rule.wts, axis=1)[:, None]

        # Step 4) Calculate p(theta_i | y, sigma^2). This a gaussian
        # approximation using the mode found in the previous optimization step.
        theta_sigma = np.sqrt(np.diagonal(-hess_inv, axis1=2, axis2=3))
        theta_mu = theta_max

        # Step 5) Calculate exceedance probabilities. We do this per sigma^2 and
        # then integrate over sigma^2
        exceedances = []
        for i in range(self.n_arms):
            exc_sigma2 = 1.0 - scipy.stats.norm.cdf(
                thresh_theta[..., None, i],
                theta_mu[..., i],
                theta_sigma[..., i],
            )
            exc = np.sum(
                exc_sigma2 * sigma2_post * self.sigma2_rule.wts[None, :], axis=1
            )
            exceedances.append(exc)
        return (sigma2_post, np.stack(exceedances, axis=-1), theta_max, theta_sigma)

    def optimize_mode(self, data, fixed_arm_dim=None, fixed_arm_values=None):
        """
        Find the mode with respect to theta of the model log joint density.

        fixed_arm_dim: we permit one of the theta arms to not be optimized: to
            be "fixed".
        fixed_arm_values: the values of the fixed arm.
        """

        # NOTE: If
        # 1) fixed_arm_values is chosen without regard to the other theta values
        # 2) sigma2 is very small
        # then, the optimization problem will be poorly conditioned and ugly because the
        # chances of t_{arm_idx} being very different from the other theta values is
        # super small with small sigma2
        # I am unsure how severe this problem is. So far, it does not appear to
        # have caused problems, but I left this comment here as a guide in case
        # the problem arises in the future.

        N = data.shape[0]
        arms_opt = list(range(self.n_arms))
        theta_max = np.zeros((N, self.sigma2_n, self.n_arms))

        if fixed_arm_dim is not None:
            arms_opt.remove(fixed_arm_dim)
            theta_max[..., fixed_arm_dim] = fixed_arm_values

        converged = False
        # The joint density is composed of:
        # 1) a quadratic term coming from the theta likelihood
        # 2) a binomial term coming from the data likelihood.
        # We ignore the terms that don't depend on theta since we are
        # optimizing here and constant offsets are irrelevant.
        for i in range(100):

            # Calculate the gradient and hessian.
            grad, hess = self.grad_hess(data, theta_max, arms_opt)
            hess_inv = np.linalg.inv(hess)

            # Take the full Newton step. The negative sign comes here because we
            # are finding a maximum, not a minimum.
            step = -np.matmul(hess_inv, grad[..., None])[..., 0]
            theta_max[..., arms_opt] += step

            # We use a step size convergence criterion. This seems empirically
            # sufficient. But, it would be possible to also check gradient norms
            # or other common convergence criteria.
            if np.max(np.linalg.norm(step, axis=-1)) < self.opt_tol:
                converged = True
                break

        if not converged:
            raise RuntimeError("Failed to identify the mode of the joint density.")

        return theta_max, hess_inv

    def log_joint(self, data, theta):
        """
        theta is expected to have shape (N, n_sigma2, n_arms)
        """
        y = data[..., 0]
        n = data[..., 1]
        theta_m0 = theta - self.mu_0
        theta_adj = theta + self.logit_p1
        exp_theta_adj = np.exp(theta_adj)
        return (
            # NB: this has fairly low accuracy in float32
            0.5 * np.einsum("...i,...ij,...j", theta_m0, self.neg_precQ, theta_m0)
            + self.logprecQdet
            + np.sum(
                theta_adj * y[:, None] - n[:, None] * np.log(exp_theta_adj + 1),
                axis=-1,
            )
            + self.log_prior
        )

    def grad_hess(self, data, theta, arms_opt):
        # These formulas are
        # straightforward derivatives from the Berry log joint density
        # see the log_joint method below
        y = data[..., 0]
        n = data[..., 1]
        na = np.arange(len(arms_opt))
        theta_m0 = theta - self.mu_0
        exp_theta_adj = np.exp(theta + self.logit_p1)
        C = 1.0 / (exp_theta_adj + 1)
        grad = (
            np.matmul(self.neg_precQ[None], theta_m0[:, :, :, None])[..., 0]
            + y[:, None]
            - (n[:, None] * exp_theta_adj) * C
        )[..., arms_opt]

        hess = np.tile(
            self.neg_precQ[None, ..., arms_opt, :][..., :, arms_opt],
            (y.shape[0], 1, 1, 1),
        )
        hess[..., na, na] -= (n[:, None] * exp_theta_adj * (C**2))[..., arms_opt]
        return grad, hess
