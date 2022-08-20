import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from lewis import batch

import outlaw.berry as berry
import outlaw.inla as inla
import outlaw.quad as quad


"""
The following class implements the Lei example.
See research/lei/lei.ipynb for the description.

We define concepts used in the code:

- `data in canonical form`:
    `data` is of shape (n_arms, 2)
    where n_arms is the number of arms in the trial and each row is a (y, n)
    pair for the corresponding arm index.
    The first row always corresponds to the control arm.

- `n configuration`:
    A valid sequence of `n` parameters in `data`
    that is observable at any point in the trial.

- `cached table`:
    A cached table is assumed to be many table of values
    row-stacked in the same order as a list of n configurations.
"""


class Lewis45:
    def __init__(
        self,
        n_arms: int,
        n_stage_1: int,
        n_stage_2: int,
        n_stage_1_interims: int,
        n_stage_1_add_per_interim: int,
        n_stage_2_add_per_interim: int,
        stage_1_futility_threshold: float,
        stage_1_efficacy_threshold: float,
        stage_2_futility_threshold: float,
        stage_2_efficacy_threshold: float,
        inter_stage_futility_threshold: float,
        posterior_difference_threshold: float,
        rejection_threshold: float,
        sig2_int=quad.log_gauss_rule(15, 2e-6, 1e3),
        n_pr_sims: int = 100,
        n_sig2_sim: int = 100,
        dtype=jnp.float64,
        cache_tables=False,
    ):
        """
        Constructs an object to run the Lei example.

        acronyms:
        - "pd" == "Posterior difference (between treatment and control arms)"
        - "pr_best" == "(Posterior) probability of best arm"
        - "pps" == "Posterior probability of success"

        Parameters:
        -----------
        n_arms:         number of arms.
        n_stage_1:      number of patients to enroll at stage 1 for each arm.
        n_stage_2:      number of patients to enroll at stage 2 for each arm.
        n_stage_1_interims:     number of interims in stage 1.
        n_stage_1_add_per_interim:      number of total patients to
                                        add per interim in stage 1.
        n_stage_2_add_per_interim:      number of patients to
                                        add in stage 2 interim to control
                                        and the selected treatment arms.
        futility_threshold:     probability cut-off to decide
                                futility for treatment arms.
                                If P(arm_i best | data) < futility_threshold,
                                declare arm_i as futile.
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
        self.n_stage_1_interims = n_stage_1_interims
        self.n_stage_1_add_per_interim = n_stage_1_add_per_interim
        self.n_stage_2_add_per_interim = n_stage_2_add_per_interim
        self.stage_1_futility_threshold = stage_1_futility_threshold
        self.stage_1_efficacy_threshold = stage_1_efficacy_threshold
        self.stage_2_futility_threshold = stage_2_futility_threshold
        self.stage_2_efficacy_threshold = stage_2_efficacy_threshold
        self.inter_stage_futility_threshold = inter_stage_futility_threshold
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
            self.n_configs_pr_best_pps_1,
            self.hashes_pr_best_pps_1,
            self.offsets_pr_best_pps_1,
            self.n_configs_pps_2,
            self.hashes_pps_2,
            self.offsets_pps_2,
            self.n_configs_pd,
            self.hashes_pd,
            self.offsets_pd,
        ) = self.n_configs_setting__()

        # diff_matrix[i]^T p = p[i+1] - p[0]
        self.diff_matrix = np.zeros((self.n_arms - 1, self.n_arms))
        self.diff_matrix[:, 0] = -1
        np.fill_diagonal(self.diff_matrix[:, 1:], 1)
        self.diff_matrix = jnp.array(self.diff_matrix)

        # order of arms used for auxiliary computations
        self.order = jnp.arange(0, self.n_arms, dtype=int)

        # posterior difference tables for every possible combination of n
        self.pr_best_table = None
        self.pps_1_table = None
        self.pps_2_table = None
        self.pd_table = None

        if cache_tables:
            self.cache_posterior_difference_table()
            self.cache_pr_best_pps_1_table()
            self.cache_pps_2_table()

    # ===============================================
    # Table caching logic
    # ===============================================

    def hash__(self, data, hashes, offsets):
        """
        Hashes `data` to return an index into a cached table
        whose table value corresponds to some function value
        using `data` as its input.
        We assume that `data` is in canonical form.

        Parameters:
        -----------
        data:   data in canonical form.
        hashes: list of sorted (increasing) hashes
                corresponding to a set of n configurations.
                Assumes that hashes was created using self.hash_n__().
                hashes[i] is the hash of the ith n configuration.
        offsets:    list of offsets into any cached tables corresponding
                    to the set of n configurations, i.e.
                    offsets[i] is the offset into a cached table
                    for the ith n configuration corresponding to hashes[i].
        """
        y = data[:, 0]
        n = data[:, 1]

        # we use the facts that:
        # - arms that are not dropped always have
        #   n value at least as large as those that were dropped.
        # - arms that are not dropped all have the same n values.
        # This means a stable sort will always:
        # - keep the first row in-place
        # - only the treatment rows will be sorted
        n_order = jnp.flip(n.shape[0] - 1 - jnp.argsort(jnp.flip(n), kind="stable"))
        n_sorted = n[n_order]
        y_sorted = y[n_order]
        n_hash = self.hash_n__(n_sorted)
        idx = jnp.searchsorted(hashes, n_hash)
        n_offset = offsets[idx]

        # this indexing works because tabling data is generated with this assumption.
        # y_sorted = [a, b, c] => corresponds to
        # relative index a*(n[-1]+1)*(n[-2]+1) + b*(n[-1]+1) + c.
        y_index = y_sorted[-1] + jnp.sum(
            jnp.flip(y_sorted[:-1]) * jnp.cumprod(jnp.flip(n_sorted[1:] + 1))
        )
        return y_index + n_offset, n_order

    def hash_n_internal__(self, n, mask):
        """
        Hashes the n configuration with a given mask.

        Parameters:
        -----------
        mask:   a mask to weigh `n` such that the weighted sum
                is unique for each n configuration.
                This should be guaranteed using self.n_configs_max_mask.
        n:      n configuration sorted in decreasing order.
        """
        return jnp.sum(mask * n)

    def hash_n__(self, n):
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
            out_all_ph2 = np.empty((0, n_arms), dtype=int)

            if n_interims <= 0:
                return out_all_ph2

            n_arr_new = np.copy(n_arr)
            for n_drop_new in range(n_drop, n_arms - 1):
                n_arr_incr = n_add // (n_arms - n_drop_new)
                n_arr_new[n_drop_new:] += n_arr_incr
                rest_all_ph2 = internal(n_arr_new, n_add, n_interims - 1, n_drop_new)
                out_all_ph2 = np.vstack(
                    (
                        out_all_ph2,
                        n_arr_new,
                        rest_all_ph2,
                    )
                )
                n_arr_new[n_drop_new:] -= n_arr_incr

            return out_all_ph2

        # make array of all n configurations
        n_arr = np.full(self.n_arms, self.n_stage_1, dtype=int)
        n_configs_ph2 = internal(
            n_arr, self.n_stage_1_add_per_interim, self.n_stage_1_interims, 0
        )
        n_configs_ph2 = np.vstack(
            (
                n_arr,
                n_configs_ph2,
            )
        )
        n_configs_ph2 = np.unique(n_configs_ph2, axis=0)

        n_configs_ph2 = np.fliplr(n_configs_ph2)

        n_configs_ph3 = np.copy(n_configs_ph2)
        n_configs_ph3[:, :2] += self.n_stage_2

        n_configs_pr_best_pps_1 = n_configs_ph2
        n_configs_pps_2 = n_configs_ph3
        n_configs_pd = np.copy(n_configs_ph3)
        n_configs_pd[:, :2] += self.n_stage_2_add_per_interim

        return n_configs_pr_best_pps_1, n_configs_pps_2, n_configs_pd

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
        (
            n_configs_pr_best_pps_1,
            n_configs_pps_2,
            n_configs_pd,
        ) = self.make_n_configs__()

        n_configs_max = jnp.max(n_configs_pd)
        n_configs_max_mask = n_configs_max ** jnp.arange(0, self.n_arms)
        n_configs_max_mask = n_configs_max_mask.astype(int)

        def settings(n_configs):
            hashes = jnp.array(
                [self.hash_n_internal__(ns, n_configs_max_mask) for ns in n_configs]
            )
            hashes_order = jnp.argsort(hashes)
            n_configs = n_configs[hashes_order]
            hashes = hashes[hashes_order]
            sizes = jnp.array([0] + [jnp.prod(ns + 1) for ns in n_configs])
            offsets = jnp.cumsum(sizes)[:-1]
            return n_configs, hashes, offsets

        return (
            n_configs_max_mask,
            *settings(n_configs_pr_best_pps_1),
            *settings(n_configs_pps_2),
            *settings(n_configs_pd),
        )

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
            process_batch__(ns, internal_jit, batch_size) for ns in self.n_configs_pd
        )
        return jnp.row_stack(tup_tables)

    def cache_posterior_difference_table(self, batch_size):
        self.pd_table = self.posterior_difference_table__(batch_size)

    def pr_best_pps_1_table__(self, key, batch_size):
        unifs = jax.random.uniform(
            key=key,
            shape=(
                self.n_pr_sims,
                self.n_stage_2 + self.n_stage_2_add_per_interim,
                self.n_arms,
            ),
        )
        _, key = jax.random.split(key)
        unifs_sig2 = jax.random.uniform(
            key=key,
            shape=(self.n_pr_sims,),
        )
        _, key = jax.random.split(key)
        normals = jax.random.normal(key, shape=(self.n_pr_sims, self.n_arms))

        def internal(data):
            return jax.vmap(self.pr_best_pps_1, in_axes=(0, None, None, None))(
                data, normals, unifs_sig2, unifs
            )

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
            process_batch__(ns, internal_jit, batch_size)
            for ns in self.n_configs_pr_best_pps_1
        )
        pr_best_tables = tuple(t[0] for t in tup_tables)
        pps_tables = tuple(t[1] for t in tup_tables)
        return jnp.row_stack(pr_best_tables), jnp.row_stack(pps_tables)

    def cache_pr_best_pps_1_table(self, key, batch_size):
        (
            self.pr_best_table,
            self.pps_1_table,
        ) = self.pr_best_pps_1_table__(key, batch_size)

    def pps_2_table__(self, key, batch_size):
        unifs = jax.random.uniform(
            key=key,
            shape=(
                self.n_pr_sims,
                self.n_stage_2_add_per_interim,
                self.n_arms,
            ),
        )
        _, key = jax.random.split(key)
        unifs_sig2 = jax.random.uniform(
            key=key,
            shape=(self.n_pr_sims,),
        )
        _, key = jax.random.split(key)
        normals = jax.random.normal(
            key=key,
            shape=(self.n_pr_sims, self.n_arms),
        )

        def internal(data):
            return jax.vmap(self.pps_2, in_axes=(0, None, None, None))(
                data, normals, unifs_sig2, unifs
            )

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
            process_batch__(ns, internal_jit, batch_size) for ns in self.n_configs_pps_2
        )
        return jnp.row_stack(tup_tables)

    def cache_pps_2_table(self, key, batch_size):
        self.pps_2_table = self.pps_2_table__(key, batch_size)

    def get_posterior_difference__(self, data):
        h, n_order = self.hash__(data, self.hashes_pd, self.offsets_pd)
        n_order_inverse = jnp.argsort(n_order)[1:] - 1
        return self.pd_table[h][n_order_inverse]

    def get_pr_best_pps_1__(self, data):
        h, n_order = self.hash__(
            data, self.hashes_pr_best_pps_1, self.offsets_pr_best_pps_1
        )
        n_order_inverse = jnp.argsort(n_order)[1:] - 1
        return (
            self.pr_best_table[h][n_order_inverse],
            self.pps_1_table[h][n_order_inverse],
        )

    def get_pps_2__(self, data):
        h, n_order = self.hash__(data, self.hashes_pps_2, self.offsets_pps_2)
        n_order_inverse = jnp.argsort(n_order)[1:] - 1
        return self.pps_2_table[h][n_order_inverse]

    # ===============================================
    # Core routines for computing Bayesian quantities
    # ===============================================

    def sample_posterior_sigma_sq(self, post, unifs):
        """
        Samples from p(sigma^2 | data) given by the density
        (up to a constant), post.
        Assumes that post is computed on the grid self.sig2_sim in the same order.
        The sampling is approximate as it samples from the discrete
        measure defined by normalizing the histogram given by post.
        """
        dFx = post[:-1] * self.dsig2_sim
        Fx = jnp.cumsum(dFx)
        Fx /= Fx[-1]
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
        sig2 = self.sig2_sim
        n_sig2 = sig2.shape[0]
        p_pinned = dict(sig2=sig2, theta=None)
        logpost, x_max, hess, iters = self.custom_ops_sim.laplace_logpost(
            np.zeros((n_sig2, n_arms), dtype=self.dtype), p_pinned, data
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
            scale = jnp.sqrt(jnp.maximum(var, 0))
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
        compute_best = jax.vmap(lambda i: self.order == i)
        return jnp.mean(compute_best(x_argmax), axis=0)

    def pps(self, data, thetas, unifs):
        # estimate P(A_i | y, n, theta_0, theta_i)
        def simulate_Ai(data, arm, new_data):
            new_data = jnp.where(
                self.diff_matrix[arm].reshape((new_data.shape[0], -1)), new_data, 0
            )
            # pool outcomes for each arm
            data = data + new_data

            return self.get_posterior_difference__(data)[arm] < self.rejection_threshold

        # compute p from logit space
        p_samples = jax.scipy.special.expit(thetas)
        berns = unifs < p_samples[:, None]
        binoms = jnp.sum(berns, axis=1)
        n_arr = jnp.full_like(binoms, unifs.shape[1])
        new_data = jnp.stack((binoms, n_arr), axis=-1)

        simulate_Ai_vmapped = jax.vmap(
            jax.vmap(simulate_Ai, in_axes=(None, 0, None)),
            in_axes=(None, None, 0),
        )
        Ai_indicators = simulate_Ai_vmapped(
            data,
            self.order[:-1],
            new_data,
        )
        out = jnp.mean(Ai_indicators, axis=0)
        return out

    def pr_best_pps_common(self, data, normals, unifs):
        # compute p(sigma^2 | y, n), mode, hessian for simulation
        # p(sigma^2 | y, n) is up to a constant
        post, x_max, hess, _ = self.posterior_sigma_sq_sim(data)

        # compute covariance of theta | data, sigma^2 for each value of self.sig2_sim.
        cov = self.hessian_to_covariance(hess)
        chol = jnp.linalg.cholesky(cov)

        # sample from p(sigma^2 | data) by getting the indices of self.sig2_sim.
        i_star = self.sample_posterior_sigma_sq(post, unifs)

        # sample theta from p(theta | data, sigma^2) given each sigma^2 from i_star.
        mean_sub = x_max[i_star]
        chol_sub = chol[i_star]
        thetas = (
            jax.vmap(lambda chol, n: chol @ n, in_axes=(0, 0))(chol_sub, normals)
            + mean_sub
        )

        return thetas

    def pr_best_pps_1(self, data, normals, unifs_sig2, unifs):
        thetas = self.pr_best_pps_common(data, normals, unifs_sig2)
        pr_best_out = self.pr_best(thetas)[1:]
        pps_out = self.pps(data, thetas, unifs)
        return pr_best_out, pps_out

    def pps_2(self, data, normals, unifs_sig2, unifs):
        thetas = self.pr_best_pps_common(data, normals, unifs_sig2)
        pps_out = self.pps(data, thetas, unifs)
        return pps_out

    # ===========
    # Trial Logic
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
        n_interims = self.n_stage_1_interims
        n_add_per_interim = self.n_stage_1_add_per_interim
        futility_threshold = self.stage_1_futility_threshold
        efficacy_threshold = self.stage_1_efficacy_threshold

        # create initial data
        n_arr = jnp.full(shape=n_arms, fill_value=n_stage_1)
        data = dist.Binomial(total_count=n_arr, probs=p).sample(key)
        data = jnp.stack((data, n_arr), axis=-1)

        # auxiliary variables
        non_dropped_idx = jnp.ones(n_arms - 1, dtype=bool)
        pr_best, pps = self.get_pr_best_pps_1__(data)

        # Stage 1:
        def body_func(args):
            i, _, _, data, _, non_dropped_idx, pr_best, pps, key = args

            # get next non-dropped indices
            non_dropped_idx = (pr_best >= futility_threshold) * non_dropped_idx
            n_non_dropped = jnp.sum(non_dropped_idx)

            # check for futility
            early_exit_futility = n_non_dropped == 0

            # check for efficacy
            n_effective = jnp.sum(pps > efficacy_threshold)
            early_exit_efficacy = n_effective > 0

            # evenly distribute the next patients across non-dropped arms
            # only if we are not early stopping stage 1.
            # Note: for simplicity, we remove the remainder patients.
            do_add = jnp.logical_not(early_exit_futility | early_exit_efficacy)
            add_idx = jnp.concatenate(
                (jnp.array(True)[None], non_dropped_idx), dtype=bool
            )
            add_idx = add_idx * do_add
            n_new = jnp.where(add_idx, n_add_per_interim // (n_non_dropped + 1), 0)
            _, key = jax.random.split(key)
            y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)
            data = data + jnp.stack((y_new, n_new), axis=-1)

            pr_best, pps = self.get_pr_best_pps_1__(data)

            return (
                i + 1,
                early_exit_futility,
                early_exit_efficacy,
                data,
                n_non_dropped,
                non_dropped_idx,
                pr_best,
                pps,
                key,
            )

        (
            _,
            early_exit_futility,
            _,
            data,
            _,
            non_dropped_idx,
            _,
            pps,
            key,
        ) = jax.lax.while_loop(
            lambda tup: (tup[0] < n_interims) & jnp.logical_not(tup[1] | tup[2]),
            body_func,
            (
                0,
                False,
                False,
                data,
                non_dropped_idx.shape[0],
                non_dropped_idx,
                pr_best,
                pps,
                key,
            ),
        )

        return (
            early_exit_futility,
            data,
            non_dropped_idx,
            pps,
            key,
        )

    def stage_2(
        self,
        data,
        best_arm,
        p,
        key,
    ):
        """
        Runs a single simulation of stage 2 of the Lei example.

        Parameters:
        -----------
        data:   simulated binomial data as in lei_stage_1 output.
        non_dropped_idx:        a boolean vector indicating which arm is non-futile.
        p:                      simulation grid-point.
        key:                    jax PRNG key.

        Returns:
        --------
        0 if no rejection, otherwise 1.
        """
        n_stage_2 = self.n_stage_2
        n_stage_2_add_per_interim = self.n_stage_2_add_per_interim
        pps_threshold_lower = self.stage_2_futility_threshold
        pps_threshold_upper = self.stage_2_efficacy_threshold
        rejection_threshold = self.rejection_threshold

        non_dropped_idx = (self.order == 0) | (self.order == best_arm)

        # add n_stage_2 number of patients to each
        # of the control and selected treatment arms.
        n_new = jnp.where(non_dropped_idx, n_stage_2, 0)
        y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)
        data = data + jnp.stack((y_new, n_new), axis=-1)

        pps = self.get_pps_2__(data)[best_arm - 1]

        # interim: check early-stop based on futility (lower) or efficacy (upper)
        early_exit_futility = pps < pps_threshold_lower
        early_exit_efficacy = pps > pps_threshold_upper
        early_exit = early_exit_futility | early_exit_efficacy
        early_exit_out = jnp.logical_not(early_exit_futility) | early_exit_efficacy

        _, key = jax.random.split(key)

        def final_analysis(data):
            n_new = jnp.where(non_dropped_idx, n_stage_2_add_per_interim, 0)
            y_new = dist.Binomial(total_count=n_new, probs=p).sample(key)
            data = data + jnp.stack((y_new, n_new), axis=-1)
            return (
                self.get_posterior_difference__(data)[best_arm - 1]
                < rejection_threshold
            )

        return jax.lax.cond(
            early_exit,
            lambda: early_exit_out,
            lambda: final_analysis(data),
        )

    def single_sim(self, p, key):
        """
        Runs a single simulation of both stage 1 and stage 2.

        Parameters:
        -----------
        p:      simulation grid-point.
        key:    jax PRNG key.
        """
        # TODO: temporary fix: binomial sampling requires this if n parameter
        # cannot be constant folded and p == 0 in some entries.
        p_clipped = jnp.where(p == 0, 1e-5, p)
        p_clipped = jnp.where(p_clipped == 1, 1 - 1e-5, p_clipped)

        # Stage 1:
        (early_exit_futility, data, non_dropped_idx, pps, key,) = self.stage_1(
            p=p_clipped,
            key=key,
        )

        # if early-exited because of efficacy,
        # pick the best arm based on PPS along with control.
        # otherwise, pick the best arm based on pr_best along with control.
        best_arm_info = jnp.where(non_dropped_idx, pps, -1)
        best_arm = jnp.argmax(best_arm_info) + 1
        _, key = jax.random.split(key)

        early_exit = early_exit_futility | (
            pps[best_arm - 1] < self.inter_stage_futility_threshold
        )

        # Stage 2 only if no early termination based on futility
        return jax.lax.cond(
            early_exit,
            lambda: False,
            lambda: self.stage_2(
                data=data,
                best_arm=best_arm,
                p=p_clipped,
                key=key,
            ),
        )

    def simulate_point(self, p, keys):
        single_sim_vmapped = jax.vmap(self.single_sim, in_axes=(None, 0))
        return single_sim_vmapped(p, keys)

    def simulate(self, grid_points, keys):
        simulate_point_vmapped = jax.vmap(self.simulate_point, in_axes=(0, None))
        return simulate_point_vmapped(grid_points, keys)
