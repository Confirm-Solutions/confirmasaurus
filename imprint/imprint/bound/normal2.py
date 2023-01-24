"""
Normal Tilt-Bound with unknown mean and variance (2 parameters).
Assumes multi-arm Normal with possibly different
(mean, variance) parameters in each arm,
but the sample size is assumed to be common across all arms.
"""
import jax
import jax.numpy as jnp

from . import optimizer as opt


def A(n, theta1, theta2):
    """
    Log-partition function for d-arm univariate normal distributions.
    Each arm i has sample size n[i] and natural parameter (theta1[i], theta2[i]).

    Parameters:
    -----------
    n:  a scalar or (d,)-vector of sample sizes for each arm.
    theta1:     a (d,)-array where theta1[i] is the
                1st order natural parameter of arm i.
    theta2:     a (d,)-array where theta2[i] is the
                2nd order natural parameter of arm i.
    """
    return -0.25 * jnp.sum(theta1**2 / theta2) - 0.5 * jnp.sum(
        n * jnp.log(-2 * theta2)
    )


def A_secant(n, theta1, theta2, v1, v2, q):
    """
    Numerically stable implementation of the secant of A:
        (A(theta + qv) - A(theta)) / q

    Parameters:
    ----------
    n:  a scalar or (d,)-vector of sample sizes for each arm.
    theta1:     a (d,)-array where theta1[i] is the
                1st order natural parameter of arm i.
    theta2:     a (d,)-array where theta2[i] is the
                2nd order natural parameter of arm i.
    v1:         a (d,)-array displacement on theta1.
    v2:         a (d,)-array displacement on theta2.
    q:          tilt parameter.
    """
    return (
        -0.25
        * jnp.sum(
            (theta2 * v1 * (q * v1 + 2 * theta1) - theta1**2 * v2)
            / (theta2 * (theta2 + q * v2))
        )
        - 0.5 * jnp.sum(n * jnp.log(1 + q * v2 / theta2)) / q
    )


def dA(n, theta1, theta2):
    """
    Gradient of the log-partition function A.

    Parameters:
    -----------
    n:  a scalar or (d,)-vector of sample sizes for each arm.
    theta1:     a (d,)-array where theta1[i] is the
                1st order natural parameter of arm i.
    theta2:     a (d,)-array where theta2[i] is the
                2nd order natural parameter of arm i.

    Return:
    -------
    dA1, dA2

    dA1:    a (d,)-array gradient of A with respect to each theta1[i].
    dA2:    a (d,)-array gradient of A with respect to each theta2[i].
    """
    dA1 = -0.5 * theta1 / theta2
    dA2 = dA1 * dA1 - 0.5 * n / theta2
    return dA1, dA2


class BaseTileQCPSolver:
    def __init__(self, n, m=1, M=1e7, tol=1e-5, eps=1e-6):
        self.n = n
        self.min = m
        self.max = M
        self.tol = tol
        self.shrink_factor = 1 - eps

    def _compute_max_bound(self, theta_02, v2s):
        max_v2s = jnp.max(v2s, axis=0)

        # return shrunken maximum so that the
        # maximum q results in well-defined objective.
        return jnp.minimum(
            self.max,
            jnp.min(
                jnp.where(
                    max_v2s > 0,
                    -theta_02 / max_v2s * self.shrink_factor,
                    jnp.inf,
                )
            ),
        )


class TileForwardQCPSolver(BaseTileQCPSolver):
    r"""
    Solves the following strictly quasi-convex optimization problem:
        minimize_q max_{v \in S} L_v(q)
        subject to q >= 1
    where
        L_v(q) = (psi(theta_0, v, q) - log(a)) / q - psi(theta_0, v, 1)
    """

    def obj_v(self, theta_01, theta_02, v1, v2, q, loga):
        secq = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            q,
        )
        sec1 = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            1,
        )
        return secq - loga / q - sec1

    def obj(self, theta_01, theta_02, v1s, v2s, q, loga):
        _obj_each_vmap = jax.vmap(self.obj_v, in_axes=(None, None, 0, 0, None, None))
        return jnp.max(_obj_each_vmap(theta_01, theta_02, v1s, v2s, q, loga))

    def obj_vmap(self, theta_01, theta_02, v1s, v2s, qs, loga):
        return jax.vmap(
            self.obj,
            in_axes=(None, None, None, None, 0, None),
        )(theta_01, theta_02, v1s, v2s, qs, loga)

    def solve(self, theta_01, theta_02, v1s, v2s, a, eps=1e-6):
        loga = jnp.log(a)
        max_trunc = self._compute_max_bound(theta_02, v2s)
        return jax.lax.cond(
            loga < -1e10,
            lambda: jnp.inf,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_01, theta_02, v1s, v2s, x, loga),
                self.min,
                max_trunc,
                self.tol,
            ),
        )


class TileBackwardQCPSolver(BaseTileQCPSolver):
    r"""
    Solves the following strictly quasi-convex optimization problem:
        minimize_q max_{v \in S} L_v(q)
        subject to q >= 1
    where
        L_v(q) = (q/(q-1)) * (psi(theta_0, v, q) / q - psi(theta_0, v, 1) - log(a))
    """

    def obj_v(self, theta_01, theta_02, v1, v2, q):
        secq = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            q,
        )
        sec1 = A_secant(
            self.n,
            theta_01,
            theta_02,
            v1,
            v2,
            1,
        )
        return secq - sec1

    def obj(self, theta_01, theta_02, v1s, v2s, q, loga):
        p = 1.0 / (1.0 - 1.0 / q)
        _obj_each_vmap = jax.vmap(self.obj_v, in_axes=(None, None, 0, 0, None))
        return p * (jnp.max(_obj_each_vmap(theta_01, theta_02, v1s, v2s, q)) - loga)

    def obj_vmap(self, theta_01, theta_02, v1s, v2s, qs, loga):
        return jax.vmap(
            self.obj,
            in_axes=(None, None, None, None, 0, None),
        )(theta_01, theta_02, v1s, v2s, qs, loga)

    def solve(self, theta_01, theta_02, v1s, v2s, a):
        loga = jnp.log(a)
        max_trunc = self._compute_max_bound(theta_02, v2s)
        return jax.lax.cond(
            loga < -1e10,
            lambda: jnp.inf,
            lambda: opt._simple_bisection(
                lambda x: self.obj_vmap(theta_01, theta_02, v1s, v2s, x, loga),
                self.min,
                max_trunc,
                self.tol,
            ),
        )


def tilt_bound_fwd(
    q,
    n,
    theta_01,
    theta_02,
    v1,
    v2,
    f0,
):
    expo = A_secant(n, theta_01, theta_02, v1, v2, q)
    expo = expo - A_secant(n, theta_01, theta_02, v1, v2, 1)
    return f0 ** (1 - 1 / q) * jnp.exp(expo)


def tilt_bound_fwd_tile(
    q,
    n,
    theta_01,
    theta_02,
    v1s,
    v2s,
    f0,
):
    def _expo(v1, v2):
        expo = A_secant(n, theta_01, theta_02, v1, v2, q)
        expo = expo - A_secant(n, theta_01, theta_02, v1, v2, 1)
        return expo

    max_expo = jnp.max(jax.vmap(_expo, in_axes=(0, 0))(v1s, v2s))
    return f0 ** (1 - 1 / q) * jnp.exp(max_expo)


def tilt_bound_bwd(
    q,
    n,
    theta_01,
    theta_02,
    v1,
    v2,
    alpha,
):
    def _bound(q):
        p = 1 / (1 - 1 / q)
        slope_diff = A_secant(n, theta_01, theta_02, v1, v2, q)
        slope_diff = slope_diff - A_secant(n, theta_01, theta_02, v1, v2, 1)
        return (alpha * jnp.exp(-slope_diff)) ** p

    return jax.lax.cond(
        q <= 1,
        lambda _: (alpha >= 1) + 0.0,
        _bound,
        q,
    )


def tilt_bound_bwd_tile(
    q,
    n,
    theta_01,
    theta_02,
    v1s,
    v2s,
    alpha,
):
    p = 1 / (1 - 1 / q)

    def _expo(v1s, v2s):
        slope_diff = A_secant(n, theta_01, theta_02, v1s, v2s, q)
        slope_diff = slope_diff - A_secant(n, theta_01, theta_02, v1s, v2s, 1)
        return slope_diff

    def _bound():
        max_expo = jnp.max(jax.vmap(_expo, in_axes=(0, 0))(v1s, v2s))
        return (alpha * jnp.exp(-max_expo)) ** p

    return jax.lax.cond(
        q <= 1,
        lambda: (alpha >= 1) + 0.0,
        _bound,
    )


class Normal2Bound:
    @staticmethod
    def get_backward_bound(family_params):
        n = family_params["n"]
        bwd_solver = TileBackwardQCPSolver(n)

        def backward_bound(alpha_target, theta0, vertices):
            v = vertices - theta0
            q_opt = bwd_solver.solve(theta0, v, alpha_target)
            return tilt_bound_bwd_tile(q_opt, n, theta0, v, alpha_target)

        return jax.jit(jax.vmap(backward_bound, in_axes=(None, 0, 0)))

    @staticmethod
    def get_forward_bound(family_params):
        n = family_params["n"]
        fwd_solver = TileForwardQCPSolver(n)

        def forward_bound(f0, theta0, vertices):
            vs = vertices - theta0
            q_opt = fwd_solver.solve(theta0, vs, f0)
            return tilt_bound_fwd_tile(q_opt, n, theta0, vs, f0)

        return jax.jit(jax.vmap(forward_bound))
