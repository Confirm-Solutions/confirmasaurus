import jax
import jax.numpy as jnp
import numpy as np

# TODO: refactor and combine with binomial.py and execute.py


def binomial_tuner(test_fnc):
    # @jax.jit
    def fnc(pointwise_alpha, theta_tiles, null_truth, uniform_samples):
        sim_size, n_arm_samples, n_arms = uniform_samples.shape
        n_tiles = pointwise_alpha.shape[0]

        p_tiles = jax.scipy.special.expit(theta_tiles)
        y = jnp.sum(uniform_samples[None] < p_tiles[:, None, None, :], axis=2)
        y_flat = y.reshape((-1, n_arms))
        n_flat = jnp.full_like(y_flat, n_arm_samples)
        data = jnp.stack((y_flat, n_flat), axis=-1)
        test_stat = test_fnc(data).reshape(y.shape)
        max_null_test = jnp.max(
            jnp.where(
                null_truth[:, None],
                test_stat,
                np.min(test_stat, axis=-1, keepdims=True),
            ),
            axis=-1,
        )

        cv_idx = np.floor((sim_size + 1) * pointwise_alpha).astype(int)
        partitioned_stats = np.partition(max_null_test, sim_size - cv_idx, axis=-1)
        sim_cv = partitioned_stats[np.arange(n_tiles), -cv_idx]

        # TODO: this check could be removed?
        # nrejects_max = cv_idx - 1
        # typeI_sum = np.sum(partitioned_stats > sim_cv[:, None], axis=1)
        # assert np.all(typeI_sum <= nrejects_max)

        return sim_cv

    return fnc


def chunked_tune(
    g, simulator, pointwise_alpha, sim_size, n_arm_samples, tile_chunk_size=5000
):
    n_tiles, n_params = g.theta_tiles.shape

    sim_cvs = np.zeros(n_tiles)
    n_tile_chunks = int(np.ceil(n_tiles / tile_chunk_size))

    # abstraction idea: this part could be controlled by accumulator/model?
    samples = np.random.uniform(size=(sim_size, n_arm_samples, n_params))
    for i in range(n_tile_chunks):
        tile_start = i * tile_chunk_size
        tile_end = (i + 1) * tile_chunk_size
        tile_end = min(tile_end, g.theta_tiles.shape[0])
        sim_cvs[tile_start:tile_end] = simulator(
            pointwise_alpha[tile_start:tile_end],
            g.theta_tiles[tile_start:tile_end],
            g.null_truth[tile_start:tile_end],
            samples,
        )
    return sim_cvs


def build_lookup_table(n_arms, n_arm_samples, test_fnc):
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
    test_unique = test_fnc(data)

    # 5. Invert the unique and the sort operations so that we know the rejection
    # value for every possible dataset.
    test = test_unique[inverse_unique][axis0, inverse_colsortidx]
    return test


@jax.jit
def lookup(table, y, n_arm_samples=35):
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
