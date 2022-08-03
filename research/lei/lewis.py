import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import inlaw.berry as berry
import inlaw.inla as inla
import inlaw.quad as quad


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
        sig2_rule=quad.log_gauss_rule(15, 1e-6, 1e3),
        n_pr_best_sims=100,
        dtype=jnp.float64,
    ):
        """
        Constructs an object to run the Lei example.

        Parameters:
        -----------
        n_stage_1:      number of patients to enroll at stage 1 for each arm.
        n_interims:     number of interims.
        n_add_per_interim:      number of total patients to
                                add per interim.
        futility_threshold:     probability cut-off to decide
                                futility for treatment arms.
                                If P(arm_i best | data) < futility_threshold,
                                declare arm_i as futile.
        n_stage_2:              number of patients to add for stage 2 for each arm.
        pps_threshold_lower:    threshold for checking futility:
                                PPS < pps_threshold_lower <=> futility.
        pps_threshold_upper:    threshold for checking efficacy:
                                PPS > pps_threshold_upper <=> efficacy.
        posterior_difference_threshold: threshold to compute posterior difference
                                        of selected arm p and control arm p.
        rejection_threshold:    threshold for rejection at the final analysis
                                (if reached):
                                P(p_selected_treatment_arm - p_control_arm <
                                    posterior_difference_threshold | data)
                                    < rejection_threshold
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
        """
        Computes the posterior of sigma^2 given data, p(sigma^2 | y)
        using INLA method.
        """
        n_arms, _ = data.shape
        sig2 = self.sig2_rule.pts
        n_sig2 = sig2.shape[0]
        p_pinned = dict(sig2=sig2, theta=None)
        f = self.custom_ops.laplace_logpost
        logpost, x_max, hess, iters = f(
            np.zeros((n_sig2, n_arms), dtype=self.dtype), p_pinned, data
        )
        post = inla.exp_and_normalize(logpost, self.sig2_rule.wts, axis=-1)

        return post, x_max, hess, iters

    def pr_normal_best(self, mean, cov, key, n_sims):
        """
        Estimates P[X_i > max_{j != i} X_j] where X ~ N(mean, cov) via sampling.
        """
        out_shape = (n_sims, *mean.shape[:-1])
        sims = jax.random.multivariate_normal(key, mean, cov, shape=out_shape)
        order = jnp.arange(0, mean.shape[-1])
        compute_pr_best_all = jax.vmap(
            lambda i: jnp.mean(jnp.argmax(sims, axis=-1) == i, axis=0)
        )
        return compute_pr_best_all(order)

    def compute_pr_best(self, data, non_futile_idx, key):
        n_arms, _ = data.shape
        post, x_max, hess, _ = self.posterior_sigma_sq(data)
        mean = x_max
        hess_fn = jax.vmap(
            lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1])
        )
        prec = -hess_fn(hess)  # (n_sigs, n_arms, n_arms)
        cov = jnp.linalg.inv(prec)
        pr_normal_best_out = self.pr_normal_best(
            mean, cov, key=key, n_sims=self.n_pr_best_sims
        )
        pr_best_out = jnp.matmul(pr_normal_best_out, post * self.sig2_rule.wts)
        return jnp.where(non_futile_idx == 0, jnp.nan, pr_best_out)

    @staticmethod
    def compute_pps(data):
        # TODO: fill in detail
        return 0.5

    def posterior_difference(self, data, arm, thresh):
        # TODO: p(sigma^2 | y) can be precomputed in a table?
        n_arms, _ = data.shape
        post, x_max, hess, _ = self.posterior_sigma_sq(data)

        post_weighted = self.sig2_rule.wts * post
        hess_fn = jax.vmap(
            lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1])
        )
        prec = -hess_fn(hess)  # (n_sigs, n_arms, n_arms)

        order = jnp.arange(n_arms)
        q1 = jnp.where(order == 0, -1, 0)
        q1 = jnp.where(jnp.arange(n_arms) == arm, 1, q1)

        loc = x_max @ q1
        scale = jnp.linalg.solve(prec, q1[None, :]) @ q1
        normal_term = jax.scipy.stats.norm.cdf(thresh, loc=loc, scale=scale)
        out = normal_term @ post_weighted

        return out

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

        data:           (number of arms, 2) where column 0 is the
                        simulated binomial data for each arm
                        and column 1 is the corresponding value
                        for the Binomial n parameter.
        n_non_futile:   number of non-futile treatment arms.
        non_futile_idx: vector of booleans indicating whether each arm is non-futile.
        pr_best:        vector containing probability of
                        being the best arm for each arm.
                        It is set to jnp.nan if the arm was dropped for
                        futility or if the arm is control (index 0).
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
        data = jnp.stack((data, n_arr), axis=-1)

        # auxiliary variables
        stage_1_not_done = True
        non_futile_idx = jnp.ones(n_arms, dtype=bool)
        _, key = jax.random.split(key)
        pr_best = self.compute_pr_best(data, non_futile_idx, key)
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
            # Note: for simplicity, we remove the remainder patients.
            # remainder = n_add_per_interim % n_arms
            n_new = jnp.where(continue_idx, n_add_per_interim // n_non_futile, 0)
            _, key = jax.random.split(key)
            y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)
            data = data + jnp.stack((y_new, n_new), axis=-1)

            # compute probability of best for each arm that are non-futile
            _, key = jax.random.split(key)
            pr_best = self.compute_pr_best(data, non_futile_idx, key)

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
                                Assume to be only well-defined
                                whenever non_futile_idx is True.
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

        # add n_stage_2 number of patients to each
        # of the control and selected treatment arms.
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
                self.posterior_difference(
                    data, best_arm, posterior_difference_threshold
                )
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
        # temporary fix: binomial sampling requires this if n parameter
        # cannot be constant folded and p == 0 in some entries.
        p_no_zeros = jnp.where(p == 0, 1e-5, p)

        # Stage 1:
        data, n_non_futile, non_futile_idx, pr_best, key = self.stage_1(
            p=p_no_zeros,
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
                p=p_no_zeros,
                key=key,
            ),
        )

    def simulate_point(self, p, keys):
        single_sim_vmapped = jax.vmap(self.single_sim, in_axes=(None, 0))
        return single_sim_vmapped(p, keys)

    def simulate(self, grid_points, keys):
        simulate_point_vmapped = jax.vmap(self.simulate_point, in_axes=(0, None))
        return simulate_point_vmapped(grid_points, keys)
