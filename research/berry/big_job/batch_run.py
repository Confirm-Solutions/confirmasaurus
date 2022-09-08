import berrylib.binomial as binomial
import berrylib.fast_inla as fast_inla
import berrylib.grid as berrylibgrid
import numpy as np
from scipy.special import logit

n_arm_samples = 35
seed = 10

name = "berry3d"
n_arms = 3
n_theta_1d = 24
sim_size = 50000
theta_min = -2.8
theta_max = -1.2
gridpt_chunk_size = 5000
sim_chunk_size = 50000

##
null_hypos = [
    berrylibgrid.HyperPlane(-np.identity(n_arms)[i], -logit(0.1)) for i in range(n_arms)
]
theta1d = [
    np.linspace(theta_min, theta_max, 2 * n_theta_1d + 1)[1::2] for i in range(n_arms)
]
theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
radii = np.empty(theta.shape)
for i in range(theta.shape[1]):
    radii[:, i] = 0.5 * (theta1d[i][1] - theta1d[i][0])
g = berrylibgrid.prune(berrylibgrid.build_grid(theta, radii, null_hypos))

theta = g.thetas
theta_tiles = g.thetas[g.grid_pt_idx]

fi = fast_inla.FastINLA(n_arms=n_arms)
rejection_table = binomial.build_rejection_table(
    n_arms, n_arm_samples, fi.rejection_inference
)

np.random.seed(seed)
accumulator = binomial.binomial_accumulator(
    lambda data: binomial.lookup_rejection(rejection_table, data[..., 0])
)

# Chunking improves performance dramatically for larger tile counts:
# ~6x for a 64^3 grid
typeI_sum = np.zeros(theta_tiles.shape[0])
typeI_score = np.zeros((theta_tiles.shape[0], n_arms))
n_gridpt_chunks = int(np.ceil(theta_tiles.shape[0] / gridpt_chunk_size))
n_sim_chunks = sim_size // sim_chunk_size
assert sim_size % sim_chunk_size == 0
n_sim_chunks, n_gridpt_chunks

for j in range(n_sim_chunks):
    samples = np.random.uniform(size=(sim_chunk_size, n_arm_samples, n_arms))
    for i in range(n_gridpt_chunks):
        gridpt_start = i * gridpt_chunk_size
        gridpt_end = (i + 1) * gridpt_chunk_size
        gridpt_end = min(gridpt_end, theta_tiles.shape[0])
        sum_chunk, score_chunk = accumulator(
            theta_tiles[gridpt_start:gridpt_end],
            g.null_truth[gridpt_start:gridpt_end],
            samples,
        )
        typeI_sum[gridpt_start:gridpt_end] += sum_chunk
        typeI_score[gridpt_start:gridpt_end] += score_chunk

corners = g.vertices
c_flat = corners.reshape((-1, 2))
tile_radii = g.radii[g.grid_pt_idx]
sim_sizes = np.full(g.n_tiles, sim_size)
total, d0, d0u, d1w, d1uw, d2uw = binomial.upper_bound(
    theta_tiles,
    tile_radii,
    corners,
    sim_sizes,
    n_arm_samples,
    typeI_sum,
    typeI_score,
)
bound_components = (total, d0, d0u, d1w, d1uw, d2uw)

grid_components = (theta, theta_tiles, tile_radii, corners, g.null_truth)
sim_components = (sim_sizes, typeI_sum, typeI_score)
np.save(
    f"output_{name}.npy",
    np.array([grid_components, sim_components, bound_components], dtype=object),
)
