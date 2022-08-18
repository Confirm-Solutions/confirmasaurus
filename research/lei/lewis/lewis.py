from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from lewis import batch

import inlaw.berry as berry
import inlaw.inla as inla
import inlaw.quad as quad


class Lewis45:
    def __init__(
        self,
        n_arms: int,
        n_stage_1: int,
        n_stage_2: int,
        n_interims: int,
        n_add_per_interim: int,
        futility_threshold: float,
        pps_threshold_lower: float,
        pps_threshold_upper: float,
        posterior_difference_threshold: float,
        rejection_threshold: float,
        sig2_int=quad.log_gauss_rule(15, 1e-6, 1e3),
        n_pr_sims: int = 100,
        n_sig2_sim: int = 100,
        batch_size: int = 2**16,
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
        self.n_pr_sims = n_pr_sims
        self.dtype = dtype

        # sig2 for quadrature integration
        self.sig2_int = sig2_int
        self.sig2_int.pts = self.sig2_int.pts.astype(self.dtype)
        self.sig2_int.wts = self.sig2_int.wts.astype(self.dtype)
        self.custom_ops_int = berry.optimized(self.sig2_int.pts, n_arms=n_arms).config(
            opt_tol=1e-3
        )

        # sig2 for simulation
        self.sig2_sim = 10 ** jnp.linspace(-6, 3, n_sig2_sim, dtype=self.dtype)
        self.dsig2_sim = jnp.diff(self.sig2_sim)
        self.custom_ops_sim = berry.optimized(self.sig2_sim, n_arms=self.n_arms).config(
            opt_tol=1e-3
        )

        ## cache
        # n configuration information
        (
            self.n_configs_max_mask,
            self.n_configs_ph2,
            self.n_configs_ph3,
            self.hashes_ph2,
            self.hashes_ph3,
            self.offsets_ph2,
            self.offsets_ph3,
        ) = self.n_configs_setting__()

        # diff_matrix[i]^T p = p[i+1] - p[0]
        self.diff_matrix = np.zeros((self.n_arms - 1, self.n_arms))
        self.diff_matrix[:, 0] = -1
        np.fill_diagonal(self.diff_matrix[:, 1:], 1)
        self.diff_matrix = jnp.array(self.diff_matrix)

        # order of arms used for auxiliary computations
        self.order = jnp.arange(0, self.n_arms, dtype=int)

        # posterior difference tables for every possible combination of n
        self.posterior_difference_table = None

        # TODO add more tables

    # ===============================================
    # Table caching logic
    # ===============================================

    def hash__(self, y, n, hashes, offsets):
        """
        Hashes (y, n) such that the resulting integer is an index into
        one of the cached tables where the value corresponds
        to the input (y, n).
        """
        n_hash = self.hash_n__(n)
        idx = jnp.searchsorted(hashes, n_hash)
        n_offset = offsets[idx]
        y_index = y[-1] + jnp.sum(jnp.flip(y[:-1]) * jnp.cumprod(n + 1)[:-1])
        return y_index + n_offset

    def hash_n_internal__(self, n, mask):
        return jnp.sum(mask * jnp.sort(n))

    def hash_n__(self, n):
        """
        Hashes n such that the resulting integer is a unique integer
        for any given possible n configuration.
        """
        return self.hash_n_internal__(n, self.n_configs_max_mask)

    def make_n_configs__(self):
        """
        Creates two 2-D arrays of all possible configurations of the `n`
        Binomial parameter configurations throughout the trial.
        Each row is a possible `n` configuration.
        The first array contains all possible Phase II configurations.
        The second array contains all possible Phase III configurations.
        """

        def internal(n_arr, n_add, n_interims, n_drop):
            n_arms = n_arr.shape[-1]
            out = np.empty((0, n_arms), dtype=int)

            if n_interims <= 0:
                return n_arr

            n_arr_new = np.copy(n_arr)
            for n_drop_new in range(n_drop, n_arms - 1):
                n_arr_incr = n_add // (n_arms - n_drop_new)
                n_arr_new[n_drop_new:] += n_arr_incr
                rest = internal(n_arr_new, n_add, n_interims - 1, n_drop_new)
                out = np.vstack(
                    (
                        out,
                        n_arr_new,
                        rest,
                    )
                )
                n_arr_new[n_drop_new:] -= n_arr_incr

            return out

        # make array of all n configurations
        n_arr = np.full(self.n_arms, self.n_stage_1, dtype=int)
        n_configs_ph2 = internal(n_arr, self.n_add_per_interim, self.n_interims, 0)
        n_configs_ph2 = np.vstack(
            (
                n_arr,
                n_configs_ph2,
            )
        )
        n_configs_ph2 = np.unique(n_configs_ph2, axis=0)
        n_configs_ph3 = np.copy(n_configs_ph2)
        n_configs_ph3[:, -2:] += +self.n_stage_2

        return n_configs_ph2, n_configs_ph3

    def n_configs_setting__(self):
        """
        Returns all necessary information regarding `n` configurations.

        Returns:
        --------
        n_configs_max_mask, n_configs, hashes, offsets

        n_configs_max_mask: mask of basis values to induce
                            unique hash of n configurations.
        n_configs:  an array of `n` configurations.
        hashes:     a 1-D vector containing the hashes of each `n` configuration.
                    in the same order as in n_configs.
        offsets:    a 1-D vector containing the offsets into a cache table
                    that corresponds to the beginning of the table
                    for a given `n` configuration.
        """
        n_configs_ph2, n_configs_ph3 = self.make_n_configs__()

        n_configs_max = jnp.max(n_configs_ph3)
        n_configs_max_mask = n_configs_max ** jnp.arange(0, self.n_arms)
        n_configs_max_mask = n_configs_max_mask.astype(int)

        hashes_ph2 = jnp.array(
            [self.hash_n_internal__(ns, n_configs_max_mask) for ns in n_configs_ph2]
        )
        hashes_ph3 = jnp.array(
            [self.hash_n_internal__(ns, n_configs_max_mask) for ns in n_configs_ph3]
        )

        # sort the hashes and re-order
        hashes_ph2_order = jnp.argsort(hashes_ph2)
        n_configs_ph2 = n_configs_ph2[hashes_ph2_order]
        hashes_ph2 = hashes_ph2[hashes_ph2_order]

        hashes_ph3_order = jnp.argsort(hashes_ph3)
        n_configs_ph3 = n_configs_ph3[hashes_ph3_order]
        hashes_ph3 = hashes_ph3[hashes_ph3_order]

        # compute offsets
        sizes_ph2 = jnp.array([0] + [jnp.prod(ns + 1) for ns in n_configs_ph2])
        offsets_ph2 = jnp.cumsum(sizes_ph2)[:-1]

        sizes_ph3 = jnp.array([0] + [jnp.prod(ns + 1) for ns in n_configs_ph3])
        offsets_ph3 = jnp.cumsum(sizes_ph3)[:-1]

        return (
            n_configs_max_mask,
            n_configs_ph2,
            n_configs_ph3,
            hashes_ph2,
            hashes_ph3,
            offsets_ph2,
            offsets_ph3,
        )

    @partial(jax.jit, static_argnums=(0,))
    def table_data__(self, ns, coords):
        """
        Creates a data array used to construct internal tables.

        Parameters:
        -----------
        ns:     n parameter.
        coords: result of calling jnp.meshgrid(..., indexing="ij")

        Returns:
        --------
        data used for table construction.
        """
        data = jnp.concatenate([c.flatten().reshape(-1, 1) for c in coords], axis=1)
        n_arr = jnp.full_like(data, ns)
        data = jnp.stack((data, n_arr), axis=-1)
        return data

    def posterior_difference_table__(self, batch_size):
        def internal(data):
            return jax.vmap(self.posterior_difference, in_axes=(0,))(data)

        def process_batch__(ns, f, batch_size):
            f_batched = batch.batch_all(
                f,
                batch_size,
                in_axes=(0,),
            )
            outs, n_padded = f_batched(
                self.table_data__(
                    ns, jnp.meshgrid(*(jnp.arange(0, n + 1) for n in ns), indexing="ij")
                )
            )
            out = jnp.row_stack(outs)
            return out[:(-n_padded)] if n_padded > 0 else out

        internal_jit = jax.jit(internal)
        tup_tables = tuple(
            process_batch__(ns, internal_jit, batch_size) for ns in self.n_configs_ph3
        )
        return jnp.row_stack(tup_tables)

    def pr_best_pps_table__(self, key, batch_size):
        unifs = jax.random.uniform(
            key=key,
            shape=(self.n_pr_sims, self.n_stage_2, self.n_arms),
        )
        _, key = jax.random.split(key)

        def internal(data):
            return jax.vmap(self.pr_best_pps, in_axes=(0, None, None))(data, key, unifs)

        def process_batch__(ns, f, batch_size):
            f_batched = batch.batch_all(
                f,
                batch_size,
                in_axes=(0,),
            )
            outs, n_padded = f_batched(
                self.table_data__(
                    ns, jnp.meshgrid(*(jnp.arange(0, n + 1) for n in ns), indexing="ij")
                )
            )
            pr_best_outs = tuple(t[0] for t in outs)
            pps_outs = tuple(t[1] for t in outs)
            pr_best_out = jnp.row_stack(pr_best_outs)
            pps_outs = jnp.row_stack(pps_outs)
            return (
                (pr_best_out[:(-n_padded)], pps_outs[:(-n_padded)])
                if n_padded > 0
                else (pr_best_out, pps_outs)
            )

        internal_jit = jax.jit(internal)
        tup_tables = tuple(
            process_batch__(ns, internal_jit, batch_size) for ns in self.n_configs_ph3
        )
        pr_best_tables = tuple(t[0] for t in tup_tables)
        pps_tables = tuple(t[1] for t in tup_tables)
        return jnp.row_stack(pr_best_tables), jnp.row_stack(pps_tables)

    def get_posterior_difference__(self, data):
        y = data[:, 0]
        n = data[:, 1]
        return self.posterior_difference_table[
            self.hash__(y, n, self.hashes_ph3, self.offsets_ph3)
        ]

    # ===============================================
    # Core routines for computing Bayesian quantities
    # ===============================================

    def sample_posterior_sigma_sq(self, post, key):
        """
        Samples from p(sigma^2 | data) given by the density
        (up to a constant), post.
        The sampling is approximate as it samples from the discrete
        measure defined by normalizing the histogram given by post.
        """
        dFx = post[:-1] * self.dsig2_sim
        Fx = jnp.cumsum(dFx)
        Fx /= Fx[-1]
        unifs = jax.random.uniform(key=key, shape=(self.n_pr_sims,))
        i_star = jnp.searchsorted(Fx, unifs)
        return i_star + 1

    def hessian_to_covariance(self, hess):
        """
        Computes the covariance from the Hessian
        (w.r.t. theta) of p(data, theta, sigma^2).

        Parameters:
        -----------
        hess:   tuple of (H_a, H_b) where H_a is of
                shape (..., n) and H_b is of shape (..., 1).
                The full Hessian is given by diag(H_a) + H_b 11^T.

        Returns:
        --------
        Covariance matrix of shape (..., n, n) by inverting
        and negating each term of hess.
        """
        _, n_arms = hess[0].shape
        hess_fn = jax.vmap(
            lambda h: jnp.diag(h[0]) + jnp.full(shape=(n_arms, n_arms), fill_value=h[1])
        )
        prec = -hess_fn(hess)  # (n_sigs, n_arms, n_arms)
        return jnp.linalg.inv(prec)

    def posterior_sigma_sq_int(self, data):
        """
        Computes p(sigma^2 | data) using INLA on a grid defined by self.sig2_int.pts.

        Returns:
        --------
        post:   p(sigma^2 | data) evaluated on self.sig2_int.pts.
        x_max:  mode of x -> p(data, x, sigma^2) evaluated
                for each point in self.sig2_int.pts.
        hess:   tuple of Hessian information (H_a, H_b) such that the Hessian
                is given as in hessian_to_covariance().
        iters:  number of iterations.
        """

        n_arms, _ = data.shape
        sig2 = self.sig2_int.pts
        n_sig2 = sig2.shape[0]
        p_pinned = dict(sig2=sig2, theta=None)
        f = self.custom_ops_int.laplace_logpost
        logpost, x_max, hess, iters = f(
            np.zeros((n_sig2, n_arms), dtype=self.dtype), p_pinned, data
        )
        post = inla.exp_and_normalize(logpost, self.sig2_int.wts, axis=-1)
        return post, x_max, hess, iters

    def posterior_sigma_sq_sim(self, data):
        """
        Computes p(sigma^2 | data) using INLA on a grid defined by self.sig2_sim.

        Returns:
        --------
        post:   p(sigma^2 | data) (up to a constant) evaluated on self.sig2_sim.
        x_max:  mode of x -> p(data, x, sigma^2)
                evaluated for each point in self.sig2_sim.
        hess:   tuple of Hessian information (H_a, H_b) such that
                the Hessian is given as in hessian_to_covariance().
        iters:  number of iterations.
        """
        n_arms, _ = data.shape
        p_pinned = dict(sig2=self.sig2_sim, theta=None)
        logpost, x_max, hess, iters = jax.jit(self.custom_ops_sim.laplace_logpost)(
            np.zeros((len(self.sig2_sim), n_arms)), p_pinned, data
        )
        max_logpost = jnp.max(logpost)
        max_post = jnp.exp(max_logpost)
        post = jnp.exp(logpost - max_logpost) * max_post
        return post, x_max, hess, iters

    def posterior_difference(self, data):
        """
        Computes p(p_i - p_0 < self.posterior_threshold | data)
        for i = 1,..., d-1 where d is the total number of arms.

        Returns:
        --------
        1-D array of length d-1 where the ith component is
        p(p_{i+1} - p_0 < self.posterior_threshold | data)
        """
        post, x_max, hess, _ = self.posterior_sigma_sq_int(data)

        post_weighted = self.sig2_int.wts * post
        cov = self.hessian_to_covariance(hess)

        def post_diff_given_sigma(mean, cov):
            loc = self.diff_matrix @ mean
            # var = [..., qi^T C qi, ..., ] where qi = self.diff_matrix[i]
            var = jnp.sum((self.diff_matrix @ cov) * self.diff_matrix, axis=-1)
            scale = jnp.sqrt(var)
            normal_term = jax.scipy.stats.norm.cdf(
                self.posterior_difference_threshold, loc=loc, scale=scale
            )
            return normal_term

        normal_term = jax.vmap(post_diff_given_sigma, in_axes=(0, 0))(x_max, cov)
        return post_weighted @ normal_term

    def pr_best(self, x):
        """
        Computes P[X_i > max_{j != i} X_j] for each i = 0,..., d-1
        where x is of shape (..., d).
        """
        x_argmax = jnp.argmax(x, axis=-1)
        compute_best = jax.vmap(lambda i: jnp.where(self.order == i, 1, 0))
        return jnp.mean(compute_best(x_argmax), axis=0)
        # TODO: move this else-where
        # return jnp.where(non_futile_idx == 0, jnp.nan, pr_best_out)

    def pr_best_pps(self, data, key, unifs):
        # compute p(sigma^2 | y, n), mode, hessian for simulation
        # p(sigma^2 | y, n) is up to a constant
        post, x_max, hess, _ = self.posterior_sigma_sq_sim(data)

        # compute covariance of theta | data, sigma^2 for each value of self.sig2_sim.
        cov = self.hessian_to_covariance(hess)

        # sample from p(sigma^2 | data) by getting the indices of self.sig2_sim.
        i_star = self.sample_posterior_sigma_sq(post, key)

        # sample theta from p(theta | data, sigma^2) given each sigma^2 from i_star.
        mean_sub = x_max[i_star]
        cov_sub = cov[i_star]
        _, key = jax.random.split(key)
        thetas = jax.random.multivariate_normal(key, mean_sub, cov_sub)

        # compute pr(best arm == i | data) for each i = 1,..., d-1.
        pr_best_out = self.pr_best(thetas)[1:]

        # compute pps(arm i | data) for each i = 1,..., d-1.
        _, key = jax.random.split(key)
        pps_out = self.pps(data, thetas, unifs)

        return pr_best_out, pps_out

    def pps(self, data, thetas, unifs):
        # estimate P(A_i | y, n, theta_0, theta_i)
        def simulate_Ai(data, arm, binoms):
            # add n_stage_2 number of patients to each
            # of the control and selected treatment arms.
            n_arr = jnp.full_like(binoms, self.n_stage_2)
            new_data = jnp.stack((binoms, n_arr), axis=-1)
            new_data = jnp.where(
                self.diff_matrix[arm].reshape((new_data.shape[0], -1)), new_data, 0
            )
            # pool outcomes for each arm
            data = data + new_data

            return (
                self.get_posterior_difference__(data)[arm + 1]
                < self.rejection_threshold
            )

        # compute p from logit space
        p_samples = jax.scipy.special.expit(thetas)
        berns = unifs < p_samples[:, None]
        binoms = jnp.sum(berns, axis=1)

        simulate_Ai_vmapped = jax.vmap(
            jax.vmap(simulate_Ai, in_axes=(None, 0, None)),
            in_axes=(None, None, 0),
        )
        Ai_indicators = simulate_Ai_vmapped(
            data,
            self.order[:-1],
            binoms,
        )
        out = jnp.mean(Ai_indicators, axis=0)
        return out

    # ===========
    # TODO
    # ===========

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
        non_futile_idx = jnp.ones(n_arms, dtype=bool)
        _, key = jax.random.split(key)
        pr_best = self.compute_pr_best(data, non_futile_idx, key)
        order = jnp.arange(0, len(non_futile_idx))

        # Stage 1:
        def body_func(args):
            i, _, data, _, non_futile_idx, pr_best, key = args
            # get non-futile arm indices (offset by 1 because of control arm)
            # force control arm to be non-futile
            non_futile_idx = (
                jnp.where(order == 0, True, pr_best >= futility_threshold)
                * non_futile_idx
            )

            # if no non-futile treatment arms, terminate trial
            # else if exactly 1 non-futile arm, terminate stage 1 by choosing that arm.
            # TODO: include the control and remove + 1 at line 600
            n_non_futile = jnp.sum(non_futile_idx[1:])

            # TODO: if n_non_futile == 0, early exits.

            # TODO: check PPS for all treatment arms
            # if PPS is > threshold, break early and don't add the new samples.

            # evenly distribute the next patients across non-futile arms
            # Note: for simplicity, we remove the remainder patients.
            # remainder = n_add_per_interim % n_arms
            n_new = jnp.where(
                non_futile_idx, n_add_per_interim // (n_non_futile + 1), 0
            )
            _, key = jax.random.split(key)
            y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)
            data = data + jnp.stack((y_new, n_new), axis=-1)

            # compute probability of best for each arm that are non-futile
            pr_best = self.compute_pr_best(data, non_futile_idx, key)

            return (
                i + 1,
                data,
                n_non_futile,
                non_futile_idx,
                pr_best,
                key,
            )

        _, _, data, n_non_futile, non_futile_idx, pr_best, key = jax.lax.while_loop(
            lambda tup: (tup[0] < n_interims),
            body_func,
            (0, data, n_arms, non_futile_idx, pr_best, key),
        )

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

        # TODO: move this logic to single_sim(). Check along with pps.
        # select best treatment arm based on probability of each arm being best
        # since non_futile_idx always treats control arm (index 0) as non-futile,
        # we read past it.
        pr_best_subset = jnp.where(non_futile_idx[1:], pr_best[1:], 0)
        best_arm = jnp.argmax(pr_best_subset) + 1

        # TODO: make sure that at this point, non_futile_idx contains exactly 2 1's.
        # - one in control
        # - one in treatment

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

        # TODO: if early stop bc of futility, return False. For efficacy, return True.
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
        # TODO: if early-exited because of PPS, pick the best arm based on PPS
        # along with control and move to stage 2.
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
