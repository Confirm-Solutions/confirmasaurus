# from pprint import pformat
import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

import confirm.mini_imprint.lewis_drivers as ld


# def status_report(adap, sim_sizes, bootstrap_cvs, pointwise_target_alpha):
#     overall_cv = np.min(bootstrap_cvs[:, 0])
#     sim_size_dist = {s: c for s, c in zip(*np.unique(sim_sizes, return_counts=True))}
#     total_effort = sum([c * s for s, c in sim_size_dist.items()])
#     sim_size_effort = {s: c * s / total_effort * 100 for s, c in
#     sim_size_dist.items()}
#     out = f"overall_cv: {overall_cv:.4f}"
#     out += f"\nnumber of tiles near critical: {n_critical}"
#     out += f"\n    and with loose bounds {n_loose}"
#     out += f"\nsim size distribution: \n{pformat(sim_size_dist, indent=4)}"
#     out += f"\nsim size effort %: \n{pformat(sim_size_effort, indent=4)}"
#     return out


def lamstar_histogram(bootstrap_cvs, sim_sizes, xlim=None, weighted=False):
    unique = np.unique(sim_sizes)
    if weighted:
        HH = [
            np.repeat(bootstrap_cvs[sim_sizes == K], K // np.min(unique))
            for K in unique
        ]
    else:
        HH = [bootstrap_cvs[sim_sizes == K] for K in unique]
    if xlim is None:
        xlim = [np.min(bootstrap_cvs) - 0.02, np.min(bootstrap_cvs) + 0.1]
    plt.hist(
        HH,
        stacked=True,
        bins=np.linspace(*xlim, 100),
        label=[f"K={K}" for K in np.unique(sim_sizes)],
    )
    plt.legend(fontsize=8)
    plt.xlabel("$\lambda^*$")
    plt.ylabel("number of tiles")


def eval_bound(model, g, sim_sizes, D, eval_pts):
    tree = scipy.spatial.KDTree(g.theta_tiles)
    _, idx = tree.query(eval_pts)
    unique_idx, inverse = np.unique(idx, return_inverse=True)
    typeI_sum = ld.rej_runner(
        model,
        sim_sizes[unique_idx],
        0.05,
        g.theta_tiles[unique_idx],
        g.null_truth[unique_idx],
        D.unifs,
        D.unifs_order,
    )
    typeI_sum = typeI_sum[inverse]
    typeI_err = typeI_sum / sim_sizes[idx]
    import confirm.mini_imprint.binomial as binomial

    delta = 0.01
    typeI_err, typeI_CI = binomial.zero_order_bound(
        typeI_sum, sim_sizes[idx], delta, 1.0
    )
    typeI_bound = typeI_err + typeI_CI

    import confirm.mini_imprint.bound.binomial as tiltbound

    fwd_solver = tiltbound.ForwardQCPSolver(n=model.n_arm_samples)
    theta0 = g.theta_tiles[idx]
    v = eval_pts - theta0
    q_opt = jax.vmap(fwd_solver.solve, in_axes=(0, 0, 0))(theta0, v, typeI_bound)

    bound = np.array(
        jax.vmap(tiltbound.q_holder_bound_fwd, in_axes=(0, None, 0, 0, 0))(
            q_opt, model.n_arm_samples, theta0, v, typeI_bound
        )
    )
    return bound


def build_2d_slice(g, pt, plot_dims, slicex=[-1, 1], slicey=[-1, 1], nx=100, ny=100):
    unplot_dims = list(set(range(g.d)) - set(plot_dims))
    nx = ny = 200
    xvs = np.linspace(*slicex, nx)
    yvs = np.linspace(*slicey, ny)
    slc2d = np.stack(np.meshgrid(xvs, yvs, indexing="ij"), axis=-1)
    slc_pts = np.empty((nx, ny, g.d))
    slc_pts[..., plot_dims] = slc2d
    slc_pts[..., unplot_dims] = pt[unplot_dims]
    return slc_pts
