import jax
import jax.numpy as jnp


def logistic(t):
    """
    Numerically stable implementation of log(1 + e^t).
    """
    return jnp.maximum(t, 0) + jnp.log(1 + jnp.exp(-jnp.abs(t)))


def A(n, t):
    """
    Log-partition function of a Bernoulli family with d-arms
    where arm i has n Bernoullis with logit t_i.
    """
    return n * jnp.sum(logistic(t))


def dA(n, t):
    """
    Gradient of the log-partition function A.
    """
    return n * jax.nn.sigmoid(t)


class ForwardQCPSolver:
    """
    This class optimizes the quasiconvex program:
    For a fixed value of n, theta_0, v, a,
        minimize_q L(q)
        subject to q >= 1
    where
        L(q) = (A(theta_0 + q * v) - A(theta_0) + a) / q
        A(theta) = n * log(1 + e^theta) (elementwise)
    """

    def __init__(
        self,
        n: int,
        cp_convg_tol: float = 1e-6,
        cp_max_entropy_tol: float = 1e7,
        cp_gamma_tol: float = 1e7,
        cp_max_iters: int = int(1e2),
        qcp_q0: float = 2.0,
        qcp_convg_tol: float = 1e-3,
    ):
        if n < 0:
            raise ValueError("n must be >= 0.")
        if cp_convg_tol <= 0:
            raise ValueError("cp_convg_tol must be positive.")
        if cp_max_entropy_tol <= 0:
            raise ValueError("cp_max_entropy_tol must be positive.")
        if cp_gamma_tol <= 0:
            raise ValueError("cp_gamma_tol must be positive.")
        if cp_max_iters < 0:
            raise ValueError("cp_max_iters must be non-negative.")
        if qcp_q0 < 1:
            raise ValueError("qcp_q0 must be >= 1.")
        if qcp_convg_tol <= 0:
            raise ValueError("qcp_convg_tol must be positive.")

        self.n = n
        self.cp_convg_tol = cp_convg_tol
        self.cp_max_entropy_tol = cp_max_entropy_tol
        self.cp_gamma_tol = cp_gamma_tol
        self.cp_max_iters = cp_max_iters
        self.qcp_q0 = qcp_q0
        self.qcp_convg_tol = qcp_convg_tol

    # ============================================================
    # Members for non-optimization routine.
    # ============================================================

    def objective(self, q, theta_0, v, a):
        return (self.A(theta_0 + q * v) - self.A(theta_0) + a) / q

    # ============================================================
    # Members for optimization routine.
    # ============================================================

    def A(self, t):
        """
        Log-partition function of a Bernoulli family with d-arms
        where arm i has n Bernoullis with logit t_i.
        """
        return A(self.n, t)

    def dA(self, t):
        """
        Gradient of the log-partition function A.
        """
        return dA(self.n, t)

    def phi_t(self, q, t, theta_0, v):
        """
        Computes phi_t(q) defined by
            A(theta_0 + qv) - tq
        This is the convex function that induces the level sets of L(q).
        Specifically, for any level t,
            {x : L(x) <= t} = {x : phi_t(x) <= bound}
        for some appropriate bound (see self._convex_feasible).
        """
        return self.A(theta_0 + q * v) - t * q

    def dphi_t(self, q, t, theta_0, v):
        """
        Computes dphi_t(q)/dq given by
            A'(theta_0 + qv)^T v - t
        """
        return jnp.sum(self.dA(theta_0 + q * v) * v) - t

    def transform(self, u):
        """
        Helper function used by self._convex_feasible.
        To solve the constrained optimization problem,
        we reparametrize q in terms of u by
            1 + logistic(u)
        This transformation is convex and has a bounded Lipschitz derivative.
        As a result, the new objective as a function of u is convex
        and has Lipschitz derivative.
        """
        return 1 + logistic(u)

    def dtransform(self, u):
        """
        Derivative of self.transform.
        """
        return jax.nn.sigmoid(u)

    def inv_transform(self, q):
        """
        Inverse mapping of self.transform.
        """
        return (q - 1) + jnp.log(1 - jnp.exp(1 - q))

    def _convex_feasible(
        self,
        t,
        theta_0,
        v,
        bound,
        q_hint,
    ):
        """
        Checks if there exists a feasible q such that L(q) <= t.

        It is solved by checking if phi_t(q) is feasible.
        phi_t(q) is feasible if there exists a q >= 1
        such that phi_t(q) <= bound.
        This function assumes that bound = A(theta_0) - a.

        To unconstrain the minimization problem, we parametrize q = log(1 + e^u) + 1.
        The algorithm uses the Barzilai-Borwein method,
        since the newly parametrized function is also convex and Lipschitz.

        Returns:
        --------
        (u_prev, q_prev, dphi_t_u_prev, u, q, phi_t_u, dphi_t_u, iters)

        u_prev:         previous value in the u-space.
        q_prev:         previous value in the q-space.
        dphi_t_u_prev:  previous dphi_t_u/du.
        u:              current value in the u-space.
        q:              current value in the q-space.
        phi_t_u:        current phi_t.
        dphi_t_u:       current dphi_t_u/du.
        iters:          number of iterations.
        """
        q0 = q_hint
        u0 = self.inv_transform(q0)

        # set previous state
        # make the previous sufficiently far from current.
        u_prev = u0 - jnp.maximum(1, 2 * self.cp_convg_tol)
        q_prev = self.transform(u_prev)
        dphi_t_u_prev = self.dphi_t(q_prev, t, theta_0, v) * self.dtransform(u_prev)

        # set current state
        u = u0
        q = q0
        phi_t_u = self.phi_t(q, t, theta_0, v)
        dphi_t_u = self.dphi_t(q, t, theta_0, v) * self.dtransform(u)

        iter = 0

        # returns True if it should continue descending.
        def _cond_func(args):
            (
                u_prev,
                _,
                _,
                u,
                _,
                phi_t_u,
                _,
                iter,
            ) = args
            return (
                # not feasible yet
                (phi_t_u > bound)
                &
                # no convergence yet
                (jnp.abs(u - u_prev) > self.cp_convg_tol)
                &
                # not reached max iterations yet
                (iter < self.cp_max_iters)
                &
                # not diverged off to +/- infinity yet
                (jnp.abs(u) <= self.cp_max_entropy_tol)
            )

        # routine for gradient descent
        def _body_func(args):
            (
                u_prev,
                q_prev,
                dphi_t_u_prev,
                u,
                q,
                phi_t_u,
                dphi_t_u,
                iter,
            ) = args

            # compute descent quantities
            abs_delta_dphi_t_u = jnp.abs(dphi_t_u - dphi_t_u_prev)
            abs_delta_u = jnp.abs(u - u_prev)

            # early exit if gamma is too large to be numerically stable
            early_exit = abs_delta_u > self.cp_gamma_tol * abs_delta_dphi_t_u
            gamma = jnp.where(early_exit, 0, abs_delta_u / abs_delta_dphi_t_u)

            # update previous states
            u_prev = u
            q_prev = q
            dphi_t_u_prev = dphi_t_u

            # update current states
            u = u - gamma * dphi_t_u
            q = self.transform(u)
            phi_t_u = self.phi_t(q, t, theta_0, v)
            dphi_t_u = self.dphi_t(q, t, theta_0, v) * self.dtransform(u)

            iter = iter + 1

            return (u_prev, q_prev, dphi_t_u_prev, u, q, phi_t_u, dphi_t_u, iter)

        args = (
            u_prev,
            q_prev,
            dphi_t_u_prev,
            u,
            q,
            phi_t_u,
            dphi_t_u,
            iter,
        )

        args = jax.lax.while_loop(
            _cond_func,
            _body_func,
            args,
        )

        return args

    def _solve(self, theta_0, v, a):
        """
        Finds the minimum point for the optimization problem
        for fixed theta_0, v, a.

        This function uses the golden bisection method.

        Returns:
        --------
        (lower, upper, q, q_hint)

        lower:      last lower function value below the minimum value.
        upper:      last upper function value above the minimum value.
        q:          optimal solution.
        q_hint:     last hint used in convex optimization.
        """

        # pre-compute some auxiliary quantities for reuse
        A0 = self.A(theta_0)
        bound = A0 - a

        # theoretical bounds that contain the minimum value
        lower = self.A(theta_0 + v) - A0
        upper = self.n * jnp.sum(jnp.maximum(v, 0))

        # initial optimal value
        # invariance: q achieves a value in [lower, upper].
        q = jnp.inf

        # hint for initial starting point of self._convex_feasible
        q_hint = self.qcp_q0

        # returns True if bisection should continue
        def _cond_func(args):
            (
                lower,
                upper,
                _,
                _,
            ) = args
            return (upper - lower) >= self.qcp_convg_tol

        # routine for bisection
        def _body_func(args):
            (
                lower,
                upper,
                q,
                q_hint,
            ) = args
            mid = (upper + lower) / 2
            (
                _,
                _,
                _,
                _,
                q_new,
                phi_t_u,
                _,
                _,
            ) = self._convex_feasible(mid, theta_0, v, bound, q_hint)
            is_feasible = phi_t_u <= bound
            lower, upper, q, q_hint = jax.lax.cond(
                is_feasible,
                lambda: (lower, mid, q_new, q_new),
                lambda: (mid, upper, q, q_hint),
            )
            return (lower, upper, q, q_hint)

        args = (lower, upper, q, q_hint)

        args = jax.lax.while_loop(
            _cond_func,
            _body_func,
            args,
        )

        return args

    def solve(self, theta_0, v, a):
        """
        Returns the optimal solution.
        """
        return self._solve(theta_0, v, a)[2]


def q_holder_bound(
    q,
    n,
    theta_0,
    v,
    f0,
):
    """
    Computes the optimal q-Holder bound given by:
        f0 * exp[L(q) - (A(theta_0 + v) - A(theta_0))]
    for fixed f0, n, theta_0, v,
    where L, A are as given in ForwardQCPSolver.
    """
    a = -jnp.log(f0)
    A0 = A(n, theta_0)
    expo = (A(n, theta_0 + q * v) - A0 + a) / q - (A(n, theta_0 + v) - A0)
    return f0 * jnp.exp(expo)
