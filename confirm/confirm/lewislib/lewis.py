import os
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import confirm.outlaw.berry as berry
import confirm.outlaw.inla as inla
import confirm.outlaw.quad as quad
from confirm.lewislib import batch
from confirm.lewislib.table import LinearInterpTable
from confirm.lewislib.table import LookupTable


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

- `pd`:
    Posterior difference (between treatment and control arms):
        P(p_i - p_0 < t | y, n)
- `pr_best`:
    Posterior probability of best arm:
        P(p_i = max_{j} p_j | y, n)
- `pps`:
    Posterior probability of success:
        P(Reject at stage 2 with all remaining
          patients added to control and selected arm |
            y, n,
            selected arm = i)
"""


@dataclass
class Lewis45Spec:
    """
    # TODO: use this more!
    The specification of the Lewis45 trial we are simulating.

    This class should not contain execution-related parameters.

    Args:
        n_arms: number of arms.
        n_stage_1: number of patients to enroll at stage 1 for each arm.
        n_stage_2: number of patients to enroll at stage 2 for each arm.
        n_stage_1_interims: number of interims in stage 1.
        n_stage_1_add_per_interim: number of total patients to
                                   add per interim in stage 1.
        n_stage_2_add_per_interim: number of patients to
                                   add in stage 2 interim to control
                                   and the selected treatment arms.
        futility_threshold: probability cut-off to decide
                            futility for treatment arms.
                            If P(arm_i best | data) < futility_threshold,
                            declare arm_i as futile.
        pps_threshold_lower: threshold for checking futility:
                             PPS < pps_threshold_lower <=> futility.
        pps_threshold_upper: threshold for checking efficacy:
                             PPS > pps_threshold_upper <=> efficacy.
        posterior_difference_threshold: threshold to compute posterior difference
                                        of selected arm p and control arm p.
        rejection_threshold: threshold for rejection at the final analysis
                             (if reached):
                             P(p_selected_treatment_arm - p_control_arm <
                                posterior_difference_threshold | data)
                                < rejection_threshold
                             <=> rejection.
    """

    n_arms: int
    n_stage_1: int
    n_stage_2: int
    n_stage_1_interims: int
    n_stage_1_add_per_interim: int
    n_stage_2_add_per_interim: int
    stage_1_futility_threshold: float
    stage_1_efficacy_threshold: float
    stage_2_futility_threshold: float
    stage_2_efficacy_threshold: float
    inter_stage_futility_threshold: float
    posterior_difference_threshold: float
    rejection_threshold: float


class Lewis45:
    def __init__(
        self,
        # TODO: replace with just spec
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
        # TODO: refactor to pull the tables out as a separate class. and allow passing
        # None. then move these params to the tables constructor. Also, reorder
        # the parameters at that time.
        sig2_int=quad.log_gauss_rule(15, 2e-6, 1e3),
        n_sig2_sims: int = 20,
        dtype=jnp.float64,
        cache_tables=False,
        key=None,
        n_table_pts=None,
        batch_size=None,
        n_pr_sims=None,
    ):
        """
        Constructs an object to run the Lei example.

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
        self.spec = Lewis45Spec(
            n_arms,
            n_stage_1,
            n_stage_2,
            n_stage_1_interims,
            n_stage_1_add_per_interim,
            n_stage_2_add_per_interim,
            stage_1_futility_threshold,
            stage_1_efficacy_threshold,
            stage_2_futility_threshold,
            stage_2_efficacy_threshold,
            inter_stage_futility_threshold,
            posterior_difference_threshold,
            rejection_threshold,
        )
        self.dtype = dtype

        # sig2 for quadrature integration
        self.sig2_int = sig2_int
        self.sig2_int.pts = self.sig2_int.pts.astype(self.dtype)
        self.sig2_int.wts = self.sig2_int.wts.astype(self.dtype)
        self.custom_ops_int = berry.optimized(self.sig2_int.pts, n_arms=n_arms).config(
            opt_tol=1e-3
        )

        # sig2 for simulation
        # TODO: should we pass this as a parameter to the relevant function?
        self.sig2_sim = 10 ** jnp.linspace(-6, 3, n_sig2_sims, dtype=self.dtype)
        self.dsig2_sim = jnp.diff(self.sig2_sim)
        self.custom_ops_sim = berry.optimized(self.sig2_sim, n_arms=self.n_arms).config(
            opt_tol=1e-3
        )

        ## cache
        # n configuration information
        (
            self.n_configs_pr_best_pps_1,
            self.n_configs_pps_2,
            self.n_configs_pd,
        ) = self._make_n_configs()

        # diff_matrix[i]^T p = p[i+1] - p[0]
        self.diff_matrix = np.zeros((self.n_arms - 1, self.n_arms))
        self.diff_matrix[:, 0] = -1
        np.fill_diagonal(self.diff_matrix[:, 1:], 1)
        self.diff_matrix = jnp.array(self.diff_matrix)

        # order of arms used for auxiliary computations
        self.order = jnp.arange(0, self.n_arms, dtype=int)

        # cache jitted internal functions
        self._posterior_difference_table_internal_jit = None
        self._pr_best_pps_1_internal_jit = None
        self._pps_2_internal_jit = None

        # posterior difference tables for every possible combination of n
        if cache_tables:
            self.loaded_tables = False
            if isinstance(cache_tables, str) and os.path.exists(cache_tables):
                self.loaded_tables = self.load_tables(cache_tables)
            if not self.loaded_tables:
                self.build_tables(key, n_table_pts, batch_size, n_pr_sims)
                if isinstance(cache_tables, str):
                    self.save_tables(cache_tables)

    def build_tables(self, key, n_table_pts, batch_size, n_pr_sims):
        self.pd_table = self._posterior_difference_table(
            batch_size=batch_size, n_points=n_table_pts
        )
        self.pr_best_pps_1_table = self._pr_best_pps_1_table(
            key=key,
            n_pr_sims=n_pr_sims,
            batch_size=batch_size,
            n_points=n_table_pts,
        )
        _, key = jax.random.split(key)
        self.pps_2_table = self._pps_2_table(
            key=key, n_pr_sims=n_pr_sims, batch_size=batch_size, n_points=n_table_pts
        )

    def load_tables(self, path):
        with open(path, "rb") as f:
            spec, tables = pickle.load(f)
        # TODO: currently this just checks spec equality before accepting the
        # cached table as correct. this is risky because the computational
        # parameters could've also changed. we should add those! it would be
        # nice to have a more general caching mechanism for lookup and
        # interpolation tables.
        if spec != self.spec:
            # TODO: we should log or raise a warning when ignoring cached
            # tables
            return False
        self.pd_table, self.pr_best_pps_1_table, self.pps_2_table = tables
        return True

    def save_tables(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                (
                    self.spec,
                    (self.pd_table, self.pr_best_pps_1_table, self.pps_2_table),
                ),
                f,
            )

    # ===============================================
    # Table caching logic
    # ===============================================

    def _make_canonical(self, data):
        # we use the facts that:
        # - arms that are not dropped always have
        #   n value at least as large as those that were dropped.
        # - arms that are not dropped all have the same n values.
        # This means a stable sort will always:
        # - keep the first row in-place
        # - only the treatment rows will be sorted
        n = data[:, 1]
        n_order = jnp.flip(n.shape[0] - 1 - jnp.argsort(jnp.flip(n), kind="stable"))
        data = data[n_order]
        data = jnp.stack((data[:, 0], data[:, 1] + 1), axis=-1)
        n_order_inverse = jnp.argsort(n_order)[1:] - 1
        return data, n_order_inverse

    def _make_n_configs(self):
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

    def _table_data(self, ns, coords):
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

    def _make_grid(self, ns, n_points):
        """
        Creates a 2-D array of shape (d, n_points)
        where d is n.shape[0].
        Each row is a 1-D gridding of points for each entry of n
        by creating evenly-spaced gridding from [0, n[i]).
        The gridding always includes 0 and n[i]-1.
        If n_points is greater than the min(n) it is clipped to be min(n).
        """

        def internal(n):
            n_points_clip = jnp.minimum(jnp.min(n), n_points)
            steps = (n - 1) // (n_points_clip - 1)
            n_no_end = steps * (n_points_clip - 1)
            return jnp.array(
                [
                    jnp.concatenate(
                        (jnp.arange(n_no_end[idx], step=steps[idx]), n[idx][None] - 1)
                    )
                    for idx in range(len(n))
                ]
            )

        return jnp.array([internal(n) for n in ns])

    def _posterior_difference_table(
        self,
        batch_size,
        n_points=None,
    ):
        def internal(data):
            return jax.vmap(self.posterior_difference, in_axes=(0,))(data)

        if n_points:
            grid = self._make_grid(self.n_configs_pd, n_points)

        def _process_batch(i, f, batch_size):
            f_batched = batch.batch_all(
                f,
                batch_size,
                in_axes=(0,),
            )

            if n_points:
                meshgrid = jnp.meshgrid(*grid[i], indexing="ij")
            else:
                meshgrid = jnp.meshgrid(
                    *(jnp.arange(0, n + 1) for n in self.n_configs_pd[i]), indexing="ij"
                )

            outs, n_padded = f_batched(self._table_data(self.n_configs_pd[i], meshgrid))
            out = jnp.row_stack(outs)
            return out[:(-n_padded)] if n_padded > 0 else out

        # if called for the first time, register jitted function
        if self._posterior_difference_table_internal_jit is None:
            self._posterior_difference_table_internal_jit = jax.jit(internal)

        tup_tables = tuple(
            _process_batch(i, self._posterior_difference_table_internal_jit, batch_size)
            for i in range(self.n_configs_pd.shape[0])
        )

        if n_points:
            return LinearInterpTable(
                self.n_configs_pd + 1,
                grid,
                jnp.array(tup_tables),
            )

        else:
            return LookupTable(self.n_configs_pd + 1, tup_tables)

    def _pr_best_pps_1_table(self, key, n_pr_sims, batch_size, n_points=None):
        unifs = jax.random.uniform(
            key=key,
            shape=(
                n_pr_sims,
                self.n_stage_2 + self.n_stage_2_add_per_interim,
                self.n_arms,
            ),
        )
        _, key = jax.random.split(key)
        unifs_sig2 = jax.random.uniform(
            key=key,
            shape=(n_pr_sims,),
        )
        _, key = jax.random.split(key)
        normals = jax.random.normal(key, shape=(n_pr_sims, self.n_arms))

        if n_points:
            grid = self._make_grid(self.n_configs_pr_best_pps_1, n_points)

        def internal(data):
            return jax.vmap(self.pr_best_pps_1, in_axes=(0, None, None, None))(
                data, normals, unifs_sig2, unifs
            )

        def _process_batch(i, f, batch_size):
            f_batched = batch.batch_all(
                f,
                batch_size,
                in_axes=(0,),
            )

            if n_points:
                meshgrid = jnp.meshgrid(*grid[i], indexing="ij")
            else:
                meshgrid = jnp.meshgrid(
                    *(jnp.arange(0, n + 1) for n in self.n_configs_pr_best_pps_1[i]),
                    indexing="ij",
                )

            outs, n_padded = f_batched(
                self._table_data(self.n_configs_pr_best_pps_1[i], meshgrid)
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

        # if called for the first time, register jitted function
        if self._pr_best_pps_1_internal_jit is None:
            self._pr_best_pps_1_internal_jit = jax.jit(internal)

        tup_tables = tuple(
            _process_batch(i, self._pr_best_pps_1_internal_jit, batch_size)
            for i in range(self.n_configs_pr_best_pps_1.shape[0])
        )
        pr_best_tables = tuple(t[0] for t in tup_tables)
        pps_tables = tuple(t[1] for t in tup_tables)
        if n_points:
            return LinearInterpTable(
                self.n_configs_pr_best_pps_1 + 1,
                grid,
                (jnp.array(pr_best_tables), jnp.array(pps_tables)),
            )
        else:
            return LookupTable(
                self.n_configs_pr_best_pps_1 + 1, (pr_best_tables, pps_tables)
            )

    def _pps_2_table(self, key, n_pr_sims, batch_size, n_points=None):
        unifs = jax.random.uniform(
            key=key,
            shape=(
                n_pr_sims,
                self.n_stage_2_add_per_interim,
                self.n_arms,
            ),
        )
        _, key = jax.random.split(key)
        unifs_sig2 = jax.random.uniform(
            key=key,
            shape=(n_pr_sims,),
        )
        _, key = jax.random.split(key)
        normals = jax.random.normal(
            key=key,
            shape=(n_pr_sims, self.n_arms),
        )

        if n_points:
            grid = self._make_grid(self.n_configs_pps_2, n_points)

        def internal(data):
            return jax.vmap(self.pps_2, in_axes=(0, None, None, None))(
                data, normals, unifs_sig2, unifs
            )

        def _process_batch(i, f, batch_size):
            f_batched = batch.batch_all(
                f,
                batch_size,
                in_axes=(0,),
            )

            if n_points:
                meshgrid = jnp.meshgrid(*grid[i], indexing="ij")
            else:
                meshgrid = jnp.meshgrid(
                    *(jnp.arange(0, n + 1) for n in self.n_configs_pps_2[i]),
                    indexing="ij",
                )

            outs, n_padded = f_batched(
                self._table_data(self.n_configs_pps_2[i], meshgrid)
            )
            out = jnp.row_stack(outs)
            return out[:(-n_padded)] if n_padded > 0 else out

        # if called for the first time, register jitted function
        if self._pps_2_internal_jit is None:
            self._pps_2_internal_jit = jax.jit(internal)

        tup_tables = tuple(
            _process_batch(i, self._pps_2_internal_jit, batch_size)
            for i in range(self.n_configs_pps_2.shape[0])
        )
        if n_points:
            return LinearInterpTable(
                self.n_configs_pps_2 + 1,
                grid,
                jnp.array(tup_tables),
            )
        else:
            return LookupTable(self.n_configs_pps_2 + 1, tup_tables)

    def _get_posterior_difference(self, data):
        data, n_order_inverse = self._make_canonical(data)
        return self.pd_table.at(data)[0][n_order_inverse]

    def _get_pr_best_pps_1(self, data):
        data, n_order_inverse = self._make_canonical(data)
        outs = self.pr_best_pps_1_table.at(data)
        return tuple(out[n_order_inverse] for out in outs)

    def _get_pps_2(self, data):
        data, n_order_inverse = self._make_canonical(data)
        return self.pps_2_table.at(data)[0][n_order_inverse]

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

            return self._get_posterior_difference(data)[arm] < self.rejection_threshold

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

    def unifs_shape(self):
        """
        Helper function that returns the necessary shape of
        uniform draws for a single simulation to ensure enough Binomial
        samples are guaranteed.
        """
        # the n-configs used to compute posterior difference
        # means that we've reached the very end of the simulation
        # so it's sufficient to find the max n among these n-configs.
        n_max = jnp.max(self.n_configs_pd)
        return (n_max, self.n_arms)

    def sample(self, berns, berns_order, berns_start, n_new_per_arm):
        berns_end = berns_start + n_new_per_arm
        berns_subset = jnp.where(
            ((berns_order >= berns_start) & (berns_order < berns_end))[:, None],
            berns,
            0,
        )
        n_new = jnp.full(shape=self.n_arms, fill_value=n_new_per_arm)
        y_new = jnp.sum(berns_subset, axis=0)
        data_new = jnp.stack((y_new, n_new), axis=-1)
        return (
            data_new,
            berns_end,
        )

    def score(self, data, p):
        return data[:, 0] - data[:, 1] * p

    def stage_1(self, berns, berns_order, berns_start=0):
        """
        Runs a single simulation of Stage 1 of the Lei example.

        Parameters:
        -----------
        berns:      a 2-D array of Bernoulli(p) draws of shape (n, d) where
                    n is the max number of patients to enroll
                    and d is the total number of arms.
        berns_order:            result of calling jnp.arange(0, berns.shape[0]).
                                It is made an argument to be able to reuse this array.
        berns_start:            starting row position into berns to begin accumulation.

        Returns:
        --------
        data, n_non_futile, non_futile_idx, pr_best, berns_start

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
        berns_start:    the next starting position to accumulate berns.
        """

        # aliases
        n_arms = berns.shape[1]
        n_stage_1 = self.n_stage_1
        n_interims = self.n_stage_1_interims
        n_add_per_interim = self.n_stage_1_add_per_interim
        futility_threshold = self.stage_1_futility_threshold
        efficacy_threshold = self.stage_1_efficacy_threshold

        # create initial data
        data, berns_start = self.sample(
            berns=berns,
            berns_order=berns_order,
            berns_start=berns_start,
            n_new_per_arm=n_stage_1,
        )

        # auxiliary variables
        non_dropped_idx = jnp.ones(n_arms - 1, dtype=bool)
        pr_best, pps = self._get_pr_best_pps_1(data)

        # Stage 1:
        def body_func(args):
            (
                i,
                _,
                _,
                data,
                _,
                non_dropped_idx,
                pr_best,
                pps,
                berns_start,
            ) = args

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
            n_new_per_arm = n_add_per_interim // (n_non_dropped + 1)
            data_new, berns_start = self.sample(
                berns=berns,
                berns_order=berns_order,
                berns_start=berns_start,
                n_new_per_arm=n_new_per_arm,
            )
            data_new = jnp.where(add_idx[:, None], data_new, 0)
            data = data + data_new

            pr_best, pps = self._get_pr_best_pps_1(data)

            return (
                i + 1,
                early_exit_futility,
                early_exit_efficacy,
                data,
                n_non_dropped,
                non_dropped_idx,
                pr_best,
                pps,
                berns_start,
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
            berns_start,
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
                berns_start,
            ),
        )

        return (
            early_exit_futility,
            data,
            non_dropped_idx,
            pps,
            berns_start,
        )

    def stage_2(self, data, best_arm, berns, berns_order, berns_start, p):
        """
        Runs a single simulation of stage 2 of the Lei example.

        Parameters:
        -----------
        data:                   data in canonical form.
        best_arm:               treatment arm index that is chosen for stage 2.
        berns:                  a 2-D array of Bernoulli(p) draws of shape (n, d)
                                where n is the max number of patients and
                                d is the number of arms.
        berns_order:            result of calling jnp.arange(0, berns.shape[0]).
                                It is made an argument to be able to reuse this array.
        berns_start:            start row position into berns to start accumulation.

        Returns:
        --------
        The test statistic:
            - 1 for early futility
            - 0 for early efficacy
            - posterior difference otherwise.
        """
        n_stage_2 = self.n_stage_2
        n_stage_2_add_per_interim = self.n_stage_2_add_per_interim
        pps_threshold_lower = self.stage_2_futility_threshold
        pps_threshold_upper = self.stage_2_efficacy_threshold

        non_dropped_idx = (self.order == 0) | (self.order == best_arm)

        # add n_stage_2 number of patients to each
        # of the control and selected treatment arms.
        data_new, berns_start = self.sample(
            berns=berns,
            berns_order=berns_order,
            berns_start=berns_start,
            n_new_per_arm=n_stage_2,
        )
        data_new = jnp.where(non_dropped_idx[:, None], data_new, 0)
        data = data + data_new

        pps = self._get_pps_2(data)[best_arm - 1]

        # interim: check early-stop based on futility (lower) or efficacy (upper)
        early_exit_futility = pps < pps_threshold_lower
        early_exit_efficacy = pps > pps_threshold_upper
        early_exit = early_exit_futility | early_exit_efficacy

        def final_analysis(data, berns_start):
            data_new, berns_start = self.sample(
                berns=berns,
                berns_order=berns_order,
                berns_start=berns_start,
                n_new_per_arm=n_stage_2_add_per_interim,
            )
            data_new = jnp.where(non_dropped_idx[:, None], data_new, 0)
            data = data + data_new
            test_stat = self._get_posterior_difference(data)[best_arm - 1]
            return (test_stat, best_arm, self.score(data, p))

        return jax.lax.cond(
            early_exit,
            # slightly confusing:
            # the test stat for an early exit is 0 if efficacy, 1 if futility
            lambda: (
                (pps <= pps_threshold_upper).astype(float),
                best_arm,
                self.score(data, p),
            ),
            lambda: final_analysis(data, berns_start),
        )

    def simulate(self, p, unifs, unifs_order):
        """
        Runs a single simulation of both stage 1 and stage 2.

        Parameters:
        -----------
        p:          simulation grid-point.
        unifs:      a 2-D array of uniform draws of shape (n, d) where
                    n is the max number of patients to enroll
                    and d is the total number of arms.
        unifs_order:            result of calling jnp.arange(0, unifs.shape[0]).
                                It is made an argument to be able to reuse this array.
        Returns:
        --------
        The test statistic:
            - 1 for early futility
            - 0 for early efficacy
            - posterior difference otherwise.
        """

        # construct bernoulli draws
        berns = unifs < p[None]

        # Stage 1:
        (early_exit_futility, data, non_dropped_idx, pps, berns_start) = self.stage_1(
            berns=berns,
            berns_order=unifs_order,
        )

        # if early-exited because of efficacy,
        # pick the best arm based on PPS along with control.
        # otherwise, pick the best arm based on pr_best along with control.
        best_arm_info = jnp.where(non_dropped_idx, pps, -1)
        best_arm = jnp.argmax(best_arm_info) + 1

        early_exit = early_exit_futility | (
            pps[best_arm - 1] < self.inter_stage_futility_threshold
        )

        # Stage 2 only if no early termination based on futility
        return jax.lax.cond(
            early_exit,
            lambda: (1.0, best_arm, jnp.zeros(self.n_arms)),
            lambda: self.stage_2(
                data=data,
                best_arm=best_arm,
                berns=berns,
                berns_order=unifs_order,
                berns_start=berns_start,
                p=p,
            ),
        )

    def simulate_rejection(self, p, null_truth, unifs, unifs_order):
        test_stat, best_arm, score = self.simulate(p, unifs, unifs_order)[0]
        rej = test_stat < self.rejection_threshold
        false_rej = rej * null_truth[best_arm - 1]
        return false_rej, score
