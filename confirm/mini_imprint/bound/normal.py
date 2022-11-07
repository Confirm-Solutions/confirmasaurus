import jax.numpy as jnp


class ForwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * (q-1) * s_sq * v ** 2 - log(f0) / q
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, v, f0):
        logf0 = jnp.log(f0)
        mv_sqrt = self.scale * jnp.abs(v)
        q_opt = jnp.sqrt(-2 * logf0) / mv_sqrt
        return jnp.maximum(q_opt, 1)


class BackwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * q * s_sq * v ** 2 - log(alpha) * q / (q-1)
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, v, alpha):
        mv_sqrt = self.scale * jnp.abs(v)
        return 1 + jnp.sqrt(-2 * jnp.log(alpha)) / mv_sqrt


class TileForwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * (q-1) * s_sq * max_v v ** 2 - log(f0) / q
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, vs, f0):
        logf0 = jnp.log(f0)
        mv_sqrt = self.scale * jnp.max(jnp.abs(vs))
        q_opt = jnp.sqrt(-2 * logf0) / mv_sqrt
        return jnp.maximum(q_opt, 1)


class TileBackwardQCPSolver:
    """
    Solves the minimization problem:
        0.5 * q * s_sq * max_v v ** 2 - log(alpha) * q / (q-1)
    with respect to q >= 1.
    """

    def __init__(self, scale):
        self.scale = scale

    def solve(self, vs, alpha):
        mv_sqrt = self.scale * jnp.max(jnp.abs(vs))
        return 1 + jnp.sqrt(-2 * jnp.log(alpha)) / mv_sqrt


def tilt_bound_fwd(q, scale, v, f0):
    p_inv = 1 - 1 / q
    expo = 0.5 * (q - 1) * (scale * v) ** 2
    return f0**p_inv * jnp.exp(expo)


def tilt_bound_fwd_tile(q, scale, vs, f0):
    p_inv = 1 - 1 / q
    max_expo = 0.5 * (q - 1) * (scale * jnp.max(jnp.abs(vs))) ** 2
    return f0**p_inv * jnp.exp(max_expo)


def tilt_bound_bwd(q, scale, v, alpha):
    p = 1 / (1 - 1 / q)
    expo = 0.5 * (q - 1) * (scale * v) ** 2
    return (alpha * jnp.exp(-expo)) ** p


def tilt_bound_bwd_tile(q, scale, vs, alpha):
    p = 1 / (1 - 1 / q)
    max_expo = 0.5 * (q - 1) * (scale * jnp.max(jnp.abs(vs))) ** 2
    return (alpha * jnp.exp(-max_expo)) ** p
