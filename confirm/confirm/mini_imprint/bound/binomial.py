import jax
import jax.numpy as jnp


def logistic(t):
    """
    Numerically stable implementation of log(1 + e^t).
    """
    return jnp.maximum(t, 0) + jnp.log(1 + jnp.exp(-jnp.abs(t)))


def logistic_secant(t, v, q, b):
    """
    Numerically stable implementation of the secant of logistic defined by:
        (logistic(t + q * v) - logistic(b)) / q
    defined for all t, v in R and q > 0.
    It is only numerically stable if t, b are not too large in magnitude
    and q is sufficiently away from 0.
    """
    t_div_q = t / q
    ls_1 = jnp.maximum(t_div_q + v, 0) - jnp.maximum(b, 0) / q
    ls_2 = jnp.log(1 + jnp.exp(-jnp.abs(t + q * v)))
    ls_2 = ls_2 - jnp.log(1 + jnp.exp(-jnp.abs(b)))
    ls_2 = ls_2 / q
    return ls_1 + ls_2


def A(n, t):
    """
    Log-partition function of a Bernoulli family with d-arms
    where arm i has n Bernoullis with logit t_i.
    """
    return n * jnp.sum(logistic(t))


def A_secant(n, t, v, q, b):
    """
    Numerically stable implementation of the secant of A:
        (A(t + q * v) - A(b)) / q
    """
    return n * jnp.sum(logistic_secant(t, v, q, b))


def dA(n, t):
    """
    Gradient of the log-partition function A.
    """
    return n * jax.nn.sigmoid(t)


def _bm_minimize(
    df_func,
    t,
    q0,
    tol,
    gamma_tol,
    max_iters,
    q_lower,
    q_upper,
):
    """
    Implements the Newton algorithm to optimize:
        minimize_q [t * f(q) + B(q)]
    where f(q) is the objective to minimize
    and B(q) is the barrier function for the constraint q >= 1,
    that is, B(q) = -log(q-1).

    We assume that f is convex and has Lipschitz derivative.
    The implementation is based on the Barzilai-Borwein method,
    since the objective is convex with Lipschitz derivative.

    Parameters:
    -----------
    df_func:    function that returns the derivative of f.
    t:          barrier method regularization parameter.
    q0:         initial starting point for Newton step.
    tol:        convergence tolerance.
    gamma_tol:  largest gamma tolerance.
    max_iters:  max number of Newton iterations.
    q_lower:    lower bound of q.
    q_higher:   upper bound of q.

    Returns:
    --------
    (q_prev, dL_prev, q, dL, iters)

    q_prev:         previous solution.
    dL_prev:        previous derivative of objective.
    q:              current solution.
    dL:             current derivative of objective.
    iters:          number of iterations.
    """

    def _bm_df(q, t):
        return t * df_func(q) - 1.0 / (q - 1)

    # set previous state
    # make the previous sufficiently far from current while still feasible.
    q_prev = q0 + jnp.maximum(1, 2 * tol)
    df_prev = _bm_df(q_prev, t)

    # set current state
    q = q0
    df = _bm_df(q, t)

    iter = 0

    # returns True if it should continue descending.
    def _cond_func(args):
        (
            q_prev,
            _,
            q,
            _,
            iter,
        ) = args
        return (
            # no convergence yet
            (jnp.abs(q - q_prev) > tol)
            &
            # not reached max iterations yet
            (iter < max_iters)
            &
            # strictly feasible
            (q_lower < q)
            &
            # not divergent
            (q < q_upper)
        )

    # routine for gradient descent
    def _body_func(args):
        (
            q_prev,
            df_prev,
            q,
            df,
            iter,
        ) = args

        # compute descent quantities
        abs_delta_df = jnp.abs(df - df_prev)
        abs_delta_u = jnp.abs(q - q_prev)

        # early exit if gamma is too large (hessian is close to 0)
        # TODO: is this the right behavior?
        early_exit = abs_delta_u > gamma_tol * abs_delta_df
        gamma = jnp.where(early_exit, 0, abs_delta_u / abs_delta_df)

        # update previous states
        q_prev = q
        df_prev = df

        # update current states
        # either q is still strictly feasible or forced to the boundary
        q = jnp.maximum(q - gamma * df, q_lower)
        df = jnp.where(
            q == q_lower,
            -jnp.inf,
            _bm_df(q, t),
        )

        iter = iter + 1

        return (q_prev, df_prev, q, df, iter)

    args = (
        q_prev,
        df_prev,
        q,
        df,
        iter,
    )

    args = jax.lax.while_loop(
        _cond_func,
        _body_func,
        args,
    )

    return args


def _bm_check_feasible(
    constraint_f,
    bm_minimize,
    t0,
    q0,
    mu,
    tol,
    q_lower,
    q_upper,
):
    """
    Checks if constraint_f is feasible, that is,
    there exists a q >= 1 such that constraint_f(q) <= 0.

    It implements the barrier method to solve the problem:
        find q
        such that constraint_f(q) <= 0
        subject to q >= 1

    Parameters:
    -----------
    constraint_f:   function to check if feasible.
    bm_minimize:    function that minimizes constraint_f + barrier function
                    and returns the optimal value.
    t0:             initial regularization value.
    q0:             initial starting point for barrier method.
                    Assumes that q0 is strictly feasible (q0 > 1).
    mu:             factor to increase regularization value.
    tol:            convergence tolerance.
    q_lower:        lower bound on q (usually == 1).
    q_upper:        upper bound on q.

    Returns:
    --------
    (t, q, f_q, is_feasible)
    t:      current regularization parameter.
    q:      current solution.
    f_q:    constraint_f at q.
    is_feasible:    True if constraint_f is feasible.
    """

    def _cond_func(args):
        (t, q, f_q) = args
        return (
            # current function value is above bound
            (f_q > 0.0)
            &
            # barrier method has not converged yet
            (1.0 / t >= tol)
            &
            # q is strictly feasible
            (q_lower < q)
            &
            # q is not divergent
            (q < q_upper)
        )

    def _body_func(args):
        (t, q, _) = args
        q = bm_minimize(q, t)
        t = t * mu
        f_q = constraint_f(q)
        return (t, q, f_q)

    f_q = constraint_f(q0)
    args = (t0, q0, f_q)
    args = jax.lax.while_loop(
        _cond_func,
        _body_func,
        args,
    )
    is_feasible = args[-1] <= 0.0
    return args + (is_feasible,)


def _qcp_bisect(
    lower,
    upper,
    q_init,
    q_hint,
    is_feasible_f,
    tol,
):
    """
    This function uses the golden bisection method
    to solve the quasiconvex minimization problem:
        minimize_q f(q)
        subject to q >= 1

    Parameters:
    -----------
    lower:      initial lower value for bisection method.
                The minimum value of f must be >= lower.
    upper:      initial upper value for bisection method.
                The minimum value of f must be <= upper.
    q_init:     initial starting point.
                f(q_init) must lie in the range [lower, upper].
    q_hint:     initial hint to start is_feasible_f.
    is_feasible_f:  function that takes in (t, q_hint)
                    and returns (q_new, is_feasible)
                    where q_new is a new point that achieves a value
                    below the given level, t, if is_feasible is True.
                    If is_feasible is False, q_new is undefined.
    tol:            convergence tolerance.

    Returns:
    --------
    (lower, upper, q, q_hint)

    lower:      last lower function value below the minimum value.
    upper:      last upper function value above the minimum value.
    q:          optimal solution.
    q_hint:     last hint used in convex optimization.
    """
    # returns True if bisection should continue
    def _cond_func(args):
        (
            lower,
            upper,
            _,
            _,
        ) = args
        return (upper - lower) >= tol

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
            q_new,
            is_feasible,
        ) = is_feasible_f(mid, q_hint)
        lower, upper, q, q_hint = jax.lax.cond(
            is_feasible,
            lambda: (lower, mid, q_new, q_new),
            lambda: (mid, upper, q, q_hint),
        )
        return (lower, upper, q, q_hint)

    args = (lower, upper, q_init, q_hint)

    args = jax.lax.while_loop(
        _cond_func,
        _body_func,
        args,
    )

    return args


class BaseQCPSolver:
    def __init__(
        self,
        n: int,
        bm_t0: float = 1.0,
        bm_mu: float = 10.0,
        bm_tol: float = 1e-4,
        cp_convg_tol: float = 1e-6,
        cp_gamma_tol: float = 1e7,
        cp_max_iters: int = int(1e2),
        qcp_q0: float = 2.0,
        qcp_convg_tol: float = 1e-3,
        q_lower: float = 1.0,
        q_upper: float = 1e7,
    ):
        if n < 0:
            raise ValueError("n must be >= 0.")
        if bm_t0 <= 0:
            raise ValueError("bm_t0 must be positive.")
        if bm_mu <= 1:
            raise ValueError("bm_mu must be > 1.")
        if bm_tol <= 0:
            raise ValueError("bm_tol must be positive.")
        if cp_convg_tol <= 0:
            raise ValueError("cp_convg_tol must be positive.")
        if cp_gamma_tol <= 0:
            raise ValueError("cp_gamma_tol must be positive.")
        if cp_max_iters < 0:
            raise ValueError("cp_max_iters must be non-negative.")
        if qcp_q0 < 1:
            raise ValueError("qcp_q0 must be >= 1.")
        if qcp_convg_tol <= 0:
            raise ValueError("qcp_convg_tol must be positive.")
        if q_lower < 1:
            raise ValueError("q_lower must be >= 1.")
        if q_upper <= q_lower:
            raise ValueError("q_upper must be greater than q_lower.")

        self.n = n
        self.bm_t0 = bm_t0
        self.bm_mu = bm_mu
        self.bm_tol = bm_tol
        self.cp_convg_tol = cp_convg_tol
        self.cp_gamma_tol = cp_gamma_tol
        self.cp_max_iters = cp_max_iters
        self.qcp_q0 = qcp_q0
        self.qcp_convg_tol = qcp_convg_tol
        self.q_lower = q_lower
        self.q_upper = q_upper


class ForwardQCPSolver(BaseQCPSolver):
    """
    This class optimizes the quasiconvex program:
    For a fixed value of n, theta_0, v, a,
        minimize_q L(q)
        subject to q >= 1
    where
        L(q) = (A(theta_0 + q * v) - A(theta_0) - np.log(a)) / q
        A(theta) = n * log(1 + e^theta) (elementwise)
    """

    # ============================================================
    # Members for non-optimization routine.
    # ============================================================

    def objective(self, q, theta_0, v, a):
        """
        Computes the objective function.

        Parameters:
        -----------
        q:          q-parameter.
        theta_0:    pivot point.
        v:          displacement from pivot point.
        a:          constant shift.
        """
        return self.A_secant(theta_0, v, q, theta_0) - jnp.log(a) / q

    # ============================================================
    # Members for optimization routine.
    # ============================================================

    def A(self, t):
        return A(self.n, t)

    def A_secant(self, t, v, q, b):
        return A_secant(self.n, t, v, q, b)

    def dA(self, t):
        return dA(self.n, t)

    def phi_t(self, q, t, theta_0, v):
        """
        Computes phi_t(q) defined by
            A(theta_0 + q * v) - t * q
        This is the convex function that induces the level sets of L(q).
        Specifically, for any level t,
            {x : L(x) <= t} = {x : phi_t(x) <= bound}
        for some appropriate bound (see self._bm_check_feasible).
        """
        return self.A(theta_0 + q * v) - t * q

    def dphi_t(self, q, t, theta_0, v):
        """
        Computes dphi_t(q)/dq given by
            A'(theta_0 + q * v)^T v - t
        """
        return jnp.sum(self.dA(theta_0 + q * v) * v) - t

    def _bm_minimize(self, q0, t_bm, t_bs, theta_0, v):
        """
        Solves:
            minimize_q f_t(q)
        where f_t(q) is the barrier objective.

        Parameters:
        -----------
        q0:         initial starting point for Newton step.
        t_bm:       t value on the central path of barrier method.
        t_bs:       t value for the current level of bisection method.
        theta_0:    pivot point.
        v:          displacement from theta_0.

        Returns:
        --------
        See _bm_minimize.
        """

        def _df_func(q):
            return self.dphi_t(q, t_bs, theta_0, v)

        return _bm_minimize(
            df_func=_df_func,
            t=t_bm,
            q0=q0,
            tol=self.cp_convg_tol,
            gamma_tol=self.cp_gamma_tol,
            max_iters=self.cp_max_iters,
            q_lower=self.q_lower,
            q_upper=self.q_upper,
        )

    def _bm_check_feasible(
        self,
        t_bs,
        theta_0,
        v,
        bound,
        q_hint,
    ):
        """
        Checks if there exists a feasible q such that L(q) <= t_bs =: t.
        """

        def _constraint_f(q):
            return self.phi_t(q, t_bs, theta_0, v) - bound

        def _bm_minimize(q, t):
            out = self._bm_minimize(q, t, t_bs, theta_0, v)
            return out[2]

        return _bm_check_feasible(
            constraint_f=_constraint_f,
            bm_minimize=_bm_minimize,
            t0=self.bm_t0,
            q0=q_hint,
            mu=self.bm_mu,
            tol=self.bm_tol,
            q_lower=self.q_lower,
            q_upper=self.q_upper,
        )

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
        bound = A0 + jnp.log(a)

        # theoretical bounds that contain the minimum value
        lower = self.A(theta_0 + v) - A0
        upper = self.n * jnp.sum(jnp.maximum(v, 0))

        # initial optimal value
        # invariance: q achieves a value in [lower, upper].
        q = jnp.inf

        # hint for initial starting point of self._convex_feasible
        q_hint = self.qcp_q0

        def _is_feasible_f(t, q_hint):
            out = self._bm_check_feasible(t, theta_0, v, bound, q_hint)
            return out[1], out[3]

        return _qcp_bisect(
            lower=lower,
            upper=upper,
            q_init=q,
            q_hint=q_hint,
            is_feasible_f=_is_feasible_f,
            tol=self.qcp_convg_tol,
        )

    def solve(self, theta_0, v, a):
        """
        Returns the optimal solution.
        """
        return self._solve(theta_0, v, a)[2]


class BackwardQCPSolver(BaseQCPSolver):
    """
    This class optimizes the quasiconvex program:
    For a fixed value of n, theta_0, v, a,
        minimize_q L(q)
        subject to q >= 1
    where
        L(q) = (q / (q-1)) * [
            (A(theta_0 + q * v) - A(theta_0)) / q
            - (A(theta_0 + v) - A(theta_0))
            - np.log(a)
        ]
        A(theta) = n * log(1 + e^theta) (elementwise)
    """

    # ============================================================
    # Members for non-optimization routine.
    # ============================================================

    def objective(self, q, theta_0, v, a):
        """
        Computes the objective function.

        Parameters:
        -----------
        q:          q-parameter.
        theta_0:    pivot point.
        v:          displacement from pivot point.
        a:          constant shift.
        """

        def _eval(q):
            p = 1 / (1 - 1 / q)
            slope_diff = self.A_secant(theta_0, v, q, theta_0)
            slope_diff = slope_diff - self.A_secant(theta_0, v, 1, theta_0)
            return p * (slope_diff - jnp.log(a))

        return jax.lax.cond(
            q <= 1,
            lambda _: jnp.where(a >= 1, 0, jnp.inf),
            _eval,
            q,
        )

    # ============================================================
    # Members for optimization routine.
    # ============================================================

    def A(self, t):
        return A(self.n, t)

    def A_secant(self, t, v, q, b):
        return A_secant(self.n, t, v, q, b)

    def dA(self, t):
        return dA(self.n, t)

    def phi_t(self, q, t, theta_0, v, a):
        p_inv = 1 - 1 / q
        A0 = self.A(theta_0)
        return q * (
            self.A_secant(theta_0, v, q, theta_0)
            - (self.A(theta_0 + v) - A0 + jnp.log(a))
            - t * p_inv
        )

    def dphi_t(self, q, t, theta_0, v, a):
        return (
            jnp.sum(self.dA(theta_0 + q * v) * v)
            - (self.A(theta_0 + v) - self.A(theta_0) + jnp.log(a))
            - t
        )

    def _bm_minimize(self, q0, t_bm, t_bs, theta_0, v, a):
        """
        Solves:
            minimize_q f_t(q)
        where f_t(q) is the barrier objective.

        Parameters:
        -----------
        q0:         initial starting point for Newton step.
        t_bm:       t value on the central path of barrier method.
        t_bs:       t value for the current level of bisection method.
        theta_0:    pivot point.
        v:          displacement from theta_0.

        Returns:
        --------
        See _bm_minimize.
        """

        def _df_func(q):
            return self.dphi_t(q, t_bs, theta_0, v, a)

        return _bm_minimize(
            df_func=_df_func,
            t=t_bm,
            q0=q0,
            tol=self.cp_convg_tol,
            gamma_tol=self.cp_gamma_tol,
            max_iters=self.cp_max_iters,
            q_lower=self.q_lower,
            q_upper=self.q_upper,
        )

    def _bm_check_feasible(
        self,
        t_bs,
        theta_0,
        v,
        a,
        q_hint,
    ):
        """
        Checks if there exists a feasible q such that L(q) <= t_bs =: t.
        """

        def _constraint_f(q):
            return self.phi_t(q, t_bs, theta_0, v, a)

        def _bm_minimize(q, t):
            out = self._bm_minimize(q, t, t_bs, theta_0, v, a)
            return out[2]

        return _bm_check_feasible(
            constraint_f=_constraint_f,
            bm_minimize=_bm_minimize,
            t0=self.bm_t0,
            q0=q_hint,
            mu=self.bm_mu,
            tol=self.bm_tol,
            q_lower=self.q_lower,
            q_upper=self.q_upper,
        )

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

        # theoretical bounds that contain the minimum value
        lower = -jnp.log(a)
        upper = self.n * jnp.sum(jnp.maximum(v, 0)) - (self.A(theta_0 + v) - A0) + lower

        # initial optimal value
        # invariance: q achieves a value in [lower, upper].
        q = jnp.inf

        # hint for initial starting point of self._convex_feasible
        q_hint = self.qcp_q0

        def _is_feasible_f(t, q_hint):
            out = self._bm_check_feasible(t, theta_0, v, a, q_hint)
            return out[1], out[3]

        return _qcp_bisect(
            lower=lower,
            upper=upper,
            q_init=q,
            q_hint=q_hint,
            is_feasible_f=_is_feasible_f,
            tol=self.qcp_convg_tol,
        )

    def solve(self, theta_0, v, a):
        """
        Returns the optimal solution.
        """
        return self._solve(theta_0, v, a)[2]


def q_holder_bound_fwd(
    q,
    n,
    theta_0,
    v,
    f0,
):
    """
    Computes the forward q-Holder bound given by:
        f0 * exp[L(q) - (A(theta_0 + v) - A(theta_0))]
    for fixed f0, n, theta_0, v,
    where L, A are as given in ForwardQCPSolver.

    Parameters:
    -----------
    q:      q parameter.
    n:      scalar Binomial size parameter.
    theta_0:    d-array pivot point.
    v:          d-array displacement vector.
    f0:         probability value at theta_0.
    """
    expo = A_secant(n, theta_0, v, q, theta_0)
    expo = expo - A_secant(n, theta_0, v, 1, theta_0)
    return f0 ** (1 - 1 / q) * jnp.exp(expo)


def q_holder_bound_bwd(
    q,
    n,
    theta_0,
    v,
    alpha,
):
    """
    Computes the backward q-Holder bound given by:
        exp(-L(q))
    where L(q) is as given in BackwardQCPSolver.
    The resulting value is alpha' such that
        q_holder_bound_fwd(q, n, theta_0, v, alpha') = alpha

    Parameters:
    -----------
    q:      q parameter.
    n:      scalar Binomial size parameter.
    theta_0:    d-array pivot point.
    v:          d-array displacement from pivot point.
    alpha:      target level.
    """

    def _bound(q):
        p = 1 / (1 - 1 / q)
        slope_diff = A_secant(n, theta_0, v, q, theta_0)
        slope_diff = slope_diff - A_secant(n, theta_0, v, 1, theta_0)
        return (alpha * jnp.exp(-slope_diff)) ** p

    return jax.lax.cond(
        q <= 1,
        lambda _: (alpha >= 1) + 0.0,
        _bound,
        q,
    )
