import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from . import newlib


# TODO: generalize to other families
def get_backward_bound(bound_module, family_params):
    scale = family_params.get("scale", 1.0)

    def backward_bound(alpha_target, theta0, vertices):
        v = vertices - theta0
        bwd_solver = bound_module.TileBackwardQCPSolver(scale=scale)
        q_opt = bwd_solver.solve(v, alpha_target)
        return bound_module.tilt_bound_bwd_tile(q_opt, scale, v, alpha_target)

    return jax.jit(jax.vmap(backward_bound, in_axes=(None, 0, 0)))


def get_forward_bound(bound_module, family_params):
    scale = family_params.get("scale", 1.0)

    def forward_bound(f0, theta0, vertices):
        fwd_solver = bound_module.TileForwardQCPSolver(scale=scale)
        vs = vertices - theta0
        q_opt = fwd_solver.solve(vs, f0)
        return bound_module.tilt_bound_fwd_tile(q_opt, scale, vs, f0)

    return jax.jit(jax.vmap(forward_bound))


def get_bound(model):
    family_params = model.family_params if hasattr(model, "family_params") else {}

    if model.family == "normal":
        import confirm.bound.normal as bound_module

    elif model.family == "binomial":
        raise Exception("not implemented")
    else:
        raise Exception("unknown family")

    return (
        get_forward_bound(bound_module, family_params),
        get_backward_bound(bound_module, family_params),
    )


class Driver:
    def __init__(self, model):
        self.model = model
        self.forward_boundv, self.backward_boundv = get_bound(model)

    def stats(self, g):
        def helper(K_df):
            K = K_df["K"].iloc[0]
            K_g = newlib.Grid(g.d, K_df, g.null_hypos)
            theta = g.get_theta()
            stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
            return stats

        return g.df.groupby("K", group_keys=False).apply(helper)

    def rej(self, g, lam, delta=0.01):
        def helper(K_df):
            K = K_df["K"].iloc[0]
            K_g = newlib.Grid(g.d, K_df, g.null_hypos)
            theta, vertices = g.get_theta_and_vertices()

            # TODO: batching
            stats = self.model.sim_batch(0, K, theta, K_g.get_null_truth())
            TI_sum = jnp.sum(stats < lam, axis=-1)
            TI_est = TI_sum / K
            TI_cp_bound = scipy.stats.beta.ppf(1 - delta, TI_sum + 1, K - TI_sum)
            # If typeI_sum == sim_sizes, scipy.stats outputs nan. Output 0 instead
            # because there is no way to go higher than 1.0
            TI_cp_bound = np.where(np.isnan(TI_cp_bound), 0, TI_cp_bound)
            TI_bound = self.forward_boundv(TI_cp_bound, theta, vertices)

            return K_df.assign(
                TI_sum=TI_sum, TI_est=TI_est, TI_cp_bound=TI_cp_bound, TI_bound=TI_bound
            )

        return g.update_data(g.df.groupby("K", group_keys=False).apply(helper))
