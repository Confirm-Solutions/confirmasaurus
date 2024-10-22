from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def binomial_accumulator(rejection_fnc):
    """
    NOTE: THIS IS DEPRECATED BECAUSE IT DOESN'T TAKE A CRITICAL VALUE AS A
    PARAMETER.

    A simple re-implementation of accumulation. This is useful for distilling
    what is happening during accumulation down to a simple linear sequence of
    operations. Retaining this could be useful for tutorials or conceptual
    introductions to Imprint since we can explain this code without introducing
    most of the framework.

    NOTE: to implement the early stopping procedure from Berry, we will need to
    change all the steps. This function is only valid for a trial with a single
    final analysis.

    theta_tiles: (n_tiles, n_arms), the logit-space parameters for each tile.
    is_null_per_arm: (n_tiles, n_arms), whether each arm's parameter is within
      the null space.
    uniform_samples: (sim_size, n_arm_samples, n_arms), uniform [0, 1] samples
      used to evaluate binomial count samples.
    """

    # We wrap and return this function since rejection_fnc needs to be known at
    # jit time.
    @jax.jit
    def fnc(theta_tiles, is_null_per_arm, uniform_samples):
        sim_size, n_arm_samples, n_arms = uniform_samples.shape

        # 1. Calculate the binomial count data.
        # The sufficient statistic for binomial is just the number of uniform draws
        # above the threshold probability. But the `p_tiles` array has shape (n_tiles,
        # n_arms). So, we add empty dimensions to broadcast and then sum across
        # n_arm_samples to produce an output `y` array of shape: (n_tiles,
        # sim_size, n_arms)

        p_tiles = jax.scipy.special.expit(theta_tiles)
        y = jnp.sum(uniform_samples[None] < p_tiles[:, None, None, :], axis=2)

        # 2. Determine if we rejected each simulated sample.
        # rejection_fnc expects inputs of shape (n, n_arms) so we must flatten
        # our 3D arrays. We reshape exceedance afterwards to bring it back to 3D
        # (n_tiles, sim_size, n_arms)
        y_flat = y.reshape((-1, n_arms))
        n_flat = jnp.full_like(y_flat, n_arm_samples)
        data = jnp.stack((y_flat, n_flat), axis=-1)
        did_reject = rejection_fnc(data).reshape(y.shape)

        # 3. Determine type I family wise error rate.
        #  a. type I is only possible when the null hypothesis is true.
        #  b. check all null hypotheses.
        #  c. sum across all the simulations.
        false_reject = (
            did_reject
            & is_null_per_arm[
                :,
                None,
            ]
        )
        any_rejection = jnp.any(false_reject, axis=-1)
        typeI_sum = any_rejection.sum(axis=-1)

        # 4. Calculate score. The score function is the primary component of the
        #    gradient used in the bound:
        #  a. for binomial, it's just: y - n * p
        #  b. only summed when there is a rejection in the given simulation
        score = y - n_arm_samples * p_tiles[:, None, :]
        typeI_score = jnp.sum(any_rejection[:, :, None] * score, axis=1)

        return typeI_sum, typeI_score

    return fnc


def build_rejection_table(n_arms, n_arm_samples, rejection_fnc):
    """
    The Berry model generally deals with n_arm_samples <= 35. This means it is
    tractable to pre-calculate whether each dataset will reject the null because
    35^4 is a fairly manageable number. We can actually reduce the number of
    calculations because the arms are symmetric and we can run only for sorted
    datasets and then extrapolate to unsorted datasets.
    """

    # 1. Construct the n_arms-dimensional grid.
    ys = np.arange(n_arm_samples + 1)
    Ygrids = np.stack(np.meshgrid(*[ys] * n_arms, indexing="ij"), axis=-1)
    Yravel = Ygrids.reshape((-1, n_arms))

    # 2. Sort the grid arms while tracking the sorting order so that we can
    # unsort later.
    colsortidx = np.argsort(Yravel, axis=-1)
    inverse_colsortidx = np.zeros(Yravel.shape, dtype=np.int32)
    axis0 = np.arange(Yravel.shape[0])[:, None]
    inverse_colsortidx[axis0, colsortidx] = np.arange(n_arms)
    Y_colsorted = Yravel[axis0, colsortidx]

    # 3. Identify the unique datasets. In a 35^4 grid, this will be about 80k
    # datasets instead of 1.7m.
    Y_unique, inverse_unique = np.unique(Y_colsorted, axis=0, return_inverse=True)

    # 4. Compute the rejections for each unique dataset.
    N = np.full_like(Y_unique, n_arm_samples)
    data = np.stack((Y_unique, N), axis=-1)
    reject_unique = rejection_fnc(data)

    # 5. Invert the unique and the sort operations so that we know the rejection
    # value for every possible dataset.
    reject = reject_unique[inverse_unique][axis0, inverse_colsortidx]
    return reject


@jax.jit
def lookup_rejection(table, y, n_arm_samples=35):
    """
    Convert the y tuple datasets into indices and lookup from the table
    constructed by `build_rejection_table`.

    This assumes n_arm_samples is constant across arms.
    """
    n_arms = y.shape[-1]
    # Compute the strided array access. For example in 3D for y = [4,8,3], and
    # n_arm_samples=35, we'd have:
    # y_index = 4 * (36 ** 2) + 8 * (36 ** 1) + 3 * (36 ** 0)
    #         = 4 * (36 ** 2) + 8 * 36 + 3
    y_index = (y * ((n_arm_samples + 1) ** jnp.arange(n_arms)[::-1])[None, :]).sum(
        axis=-1
    )
    return table[y_index, :]


def upper_bound(
    theta_tiles,
    tile_radii,
    eval_pts,
    sim_sizes,
    n_arm_samples,
    typeI_sum,
    typeI_score,
    delta=0.025,
    delta_prop_0to1=0.5,
):
    """
    Compute the Imprint upper bound after simulations have been run.

    Note that normally eval_pts is a 3D array of shape:
    (n_tiles, n_corners_per_tile, n_arms)

    But, if you want to evaluate the bound for specific points that are not the
    corners, you can pass in a 3D array like:
    (n_tiles, 1, n_arms)
    """
    p_tiles = jax.scipy.special.expit(theta_tiles)
    v_diff = eval_pts - theta_tiles[:, None]
    v_sq = v_diff**2

    #
    # Step 1. 0th order terms.
    #
    # monte carlo estimate of type I error at each theta.
    d0, d0u = zero_order_bound(typeI_sum, sim_sizes, delta, delta_prop_0to1)

    #
    # Step 2. 1st order terms.
    #
    # Monte carlo estimate of gradient of type I error at the grid points
    # then dot product with the vector from the center to the corner.
    d1, d1u = first_order_bound(
        p_tiles,
        v_diff,
        v_sq,
        typeI_score,
        sim_sizes,
        n_arm_samples,
        delta,
        delta_prop_0to1,
    )

    #
    # Step 3. 2nd order terms.
    #
    d2u = second_order_bound(v_sq, theta_tiles, tile_radii, n_arm_samples)

    #
    # Step 4. Identify the eval_pts with the highest upper bound.
    #

    # The total of the bound component that varies between eval_pts.
    total_var = d1u + d2u + d1
    total_var = np.where(np.isnan(total_var), 0, total_var)
    worst_eval_pt = total_var.argmax(axis=1)
    ti = np.arange(d1.shape[0])
    d1w = d1[ti, worst_eval_pt]
    d1uw = d1u[ti, worst_eval_pt]
    d2uw = d2u[ti, worst_eval_pt]

    #
    # Step 5. Compute the total bound and return it
    #
    total_bound = d0 + d0u + d1w + d1uw + d2uw

    return total_bound, d0, d0u, d1w, d1uw, d2uw


def zero_order_bound(typeI_sum, sim_sizes, delta, delta_prop_0to1):
    import scipy.stats

    d0 = typeI_sum / sim_sizes
    # clopper-pearson upper bound in beta form.
    d0u_factor = 1.0 - delta * delta_prop_0to1
    # NOTE: moving this to JAX is nontrivial and probably best done with
    # tensorflow_probability:
    # https://github.com/google/jax/issues/2399#issuecomment-1225990206
    # https://github.com/pyro-ppl/numpyro/blob/e28a3feaa4f95d76b361101f0c75dcb5add2365e/numpyro/distributions/util.py#L426
    # Also, probably unnecessary since this is fast!
    d0u = scipy.stats.beta.ppf(d0u_factor, typeI_sum + 1, sim_sizes - typeI_sum) - d0
    # If typeI_sum == sim_sizes, scipy.stats outputs nan. Output 0 instead
    # because there is no way to go higher than 1.0
    d0u = jnp.where(jnp.isnan(d0u), 0, d0u)
    return d0, d0u


def first_order_bound(
    p_tiles, v_diff, v_sq, typeI_score, sim_sizes, n_arm_samples, delta, delta_prop_0to1
):
    d1 = ((typeI_score / sim_sizes[:, None])[:, None] * v_diff).sum(axis=-1)
    d1u_factor = np.sqrt(1 / ((1 - delta_prop_0to1) * delta) - 1.0)
    covar_quadform = (
        n_arm_samples * v_sq * p_tiles[:, None] * (1 - p_tiles[:, None])
    ).sum(axis=-1)
    # Upper bound on d1!
    d1u = np.sqrt(covar_quadform) * (d1u_factor / np.sqrt(sim_sizes)[:, None])
    return d1, d1u


def second_order_bound(v_sq, theta_tiles, tile_radii, n_arm_samples):
    n_eval_per_tile = v_sq.shape[1]

    p_lower = np.tile(
        jax.scipy.special.expit(theta_tiles - tile_radii)[:, None],
        (1, n_eval_per_tile, 1),
    )
    p_upper = np.tile(
        jax.scipy.special.expit(theta_tiles + tile_radii)[:, None],
        (1, n_eval_per_tile, 1),
    )
    special = (p_lower <= 0.5) & (0.5 <= p_upper)
    max_p = np.where(np.abs(p_upper - 0.5) < np.abs(p_lower - 0.5), p_upper, p_lower)
    hess_comp = np.where(special, 0.25 * v_sq, (max_p * (1 - max_p) * v_sq))

    hessian_quadform_bound = hess_comp.sum(axis=-1) * n_arm_samples
    d2u = 0.5 * hessian_quadform_bound
    return d2u


def optimal_centering(f, p):
    return 1 / (1 + ((1 - f) / f) ** (1 / (p - 1)))


constant_func_cache = {}


def _build_odi_constant_func(q: float):
    """
    Fully numerical integration constant evaluator. This can be useful for
    non-integer q.

    Args:
        q: The moment to compute. Must be a float greater than 1.
    """
    import numpyro.distributions as dist

    def f(n, p):
        if isinstance(p, float):
            pf = np.array([p])
        else:
            pf = p.flatten()
        xs = jnp.arange(n + 1).astype(jnp.float64)
        return jnp.exp(
            jax.scipy.special.logsumexp(
                q * jnp.log(jnp.abs(xs - n * pf)) + dist.Binomial(n, pf).log_prob(xs)
            )
        )

    if q not in constant_func_cache:
        constant_func_cache[q] = jax.jit(
            jax.vmap(f, in_axes=(None, 0)), static_argnums=(0,)
        )

    return constant_func_cache[q]


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _calc_Cqpp(
    theta_tiles,
    tile_corners,
    n_arm_samples: int,
    holderq: int,
    C_f: Callable,
    radius_ratio: float = 1.0,
):
    """
    Calculate C_q^{''} from the paper for a tile.

    See holder_odi_bound for more details on the arguments.

    This function is mostly model-independent. The part that would need to
    change is the identification of sign changes.
    """

    # NOTE: code to calculate roots symbolically:
    # q = 6
    # sp_p, sp_t, sp_n = sp.var("p, t, n")
    # binomial_mgf = (1 - sp_p + sp_p * sp.exp(sp_t)) ** sp_n
    # mean = sp.diff(binomial_mgf, sp_t).subs(sp_t, 0)
    # central_moment = (
    #     sp.diff(sp.exp(-mean * sp_t) * binomial_mgf, sp_t, q).subs(sp_t, 0).simplify()
    # )
    # rts = sp.roots(sp.Poly(sp.diff(central_moment.subs(n, 50))))

    # we know that the only derivative sign change is at 0.5 for q < 16
    # we know less about odd q because of the absolute value. so, for now, we
    # ban odd q and q > 16
    if not isinstance(holderq, int) or holderq > 16 or holderq % 2 != 0:
        raise ValueError("The q parameter must be an even integer less than 16.")

    holderp = 1 / (1 - 1.0 / holderq)
    sup_v = jnp.max(
        jnp.sum(
            jnp.where(
                jnp.isnan(tile_corners),
                0,
                jnp.abs(radius_ratio * (tile_corners - theta_tiles[:, None]))
                ** holderp,
            ),
            axis=2,
        )
        ** (1.0 / holderp),
        axis=1,
    )

    tile_corners_p = jax.scipy.special.expit(tile_corners)

    # NOTE: we are assuming that we know the supremum occurs at a corner or at
    # p=0.5. This might not be true for other models or for q > 16.
    C_corners = jnp.where(
        jnp.isnan(tile_corners_p.ravel()),
        0,
        C_f(n_arm_samples, tile_corners_p.ravel()),
    ).reshape(tile_corners_p.shape)

    # maximum per dimension over the corners of the tile
    C_max = jnp.max(C_corners, axis=1)
    # if the tile crosses p=0.5, we just set C_max equal to the value at 0.5
    C_max = jnp.where(
        jnp.any(tile_corners_p < 0.5, axis=1) & jnp.any(tile_corners_p > 0.5, axis=1),
        C_f(n_arm_samples, jnp.array([0.5]))[0],
        C_max,
    )

    # finally, sum across the arms to compute the full multidimensional moment
    # expectation
    sup_moment = jnp.sum(C_max, axis=-1) ** (1 / holderq)

    # Cq'' from the paper:
    return sup_v * sup_moment


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def holder_odi_bound(
    typeI_bound,
    theta_tiles,
    tile_corners,
    n_arm_samples: int,
    holderq: int,
    radius_ratio: float = 1.0,
    C_f: Callable = None,
):
    """
    Compute the Holder-ODI on Type I Error. See the paper for mathematical
    details.

    Args:
        typeI_bound: the type I error bound at the simulation points.
        theta_tiles: numpy array containing tile simulation points with shape:
            (n_tiles, n_arms)
        tile_corners: numpy array containing tile corners with shape:
            (n_tiles, n_corners_per_tile, n_arms)
        n_arm_samples: number of samples per arm, Binomial n parameter
        holderq: Holder exponent, q in C_q^{''}

    Returns:
        The Holder ODI type I error bound for each tile.
    """
    if C_f is None:
        C_f = _build_odi_constant_func(holderq)
    Cqpp = _calc_Cqpp(
        theta_tiles,
        tile_corners,
        n_arm_samples,
        holderq,
        C_f,
        radius_ratio=radius_ratio,
    )
    return (Cqpp / holderq + typeI_bound ** (1 / holderq)) ** holderq


@partial(jax.jit, static_argnums=(3, 4))
def invert_bound(bound, theta_tiles, vertices, n_arm_samples, holderq):
    C_f = _build_odi_constant_func(holderq)
    Cqpp = _calc_Cqpp(theta_tiles, vertices, n_arm_samples, holderq, C_f)
    pointwise_bound = (bound ** (1 / holderq) - Cqpp / holderq) ** holderq
    return jnp.where(pointwise_bound > bound, 0.0, pointwise_bound)
