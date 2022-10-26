from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np


def status_report(adap, sim_sizes, bootstrap_cvs, pointwise_target_alpha):
    overall_cv = np.min(bootstrap_cvs[:, 0])
    n_critical = np.sum((bootstrap_cvs[:, 0] < overall_cv + 0.01))
    n_loose = np.sum(
        (bootstrap_cvs[:, 0] < overall_cv + 0.01)
        & (adap.alpha_target - pointwise_target_alpha > adap.grid_target)
    )
    sim_size_dist = {s: c for s, c in zip(*np.unique(sim_sizes, return_counts=True))}
    total_effort = sum([c * s for s, c in sim_size_dist.items()])
    sim_size_effort = {s: c * s / total_effort * 100 for s, c in sim_size_dist.items()}
    out = f"overall_cv: {overall_cv:.4f}"
    out += f"\nnumber of tiles near critical: {n_critical}"
    out += f"\n    and with loose bounds {n_loose}"
    out += f"\nsim size distribution: \n{pformat(sim_size_dist, indent=4)}"
    out += f"\nsim size effort %: \n{pformat(sim_size_effort, indent=4)}"
    return out


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


# plt.figure(figsize=(12, 7))
# plt.suptitle(title)
# plt.subplot(1, 2, 1)
# lamstar_histogram(bootstrap_cvs[:, idx], sim_sizes)
# plt.subplot(1, 2, 2)
# lamstar_histogram(bootstrap_cvs[:, idx], sim_sizes, weighted=True)
# plt.show()
