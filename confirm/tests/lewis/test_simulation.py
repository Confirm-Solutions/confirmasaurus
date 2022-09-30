import jax
import jax.numpy as jnp

from confirm.lewislib import lewis


default_params = {
    "n_arms": 3,
    "n_stage_1": 1,
    "n_stage_2": 1,
    "n_stage_1_interims": 2,
    "n_stage_1_add_per_interim": 3,
    "n_stage_2_add_per_interim": 1,
    "stage_1_futility_threshold": 0.2,
    "stage_2_futility_threshold": 0.1,
    "stage_1_efficacy_threshold": 0.9,
    "stage_2_efficacy_threshold": 0.9,
    "inter_stage_futility_threshold": 0.8,
    "posterior_difference_threshold": 0.05,
    "rejection_threshold": 0.05,
    "batch_size": 2**16,
    "key": jax.random.PRNGKey(1),
    "n_pr_sims": 100,
    "n_sig2_sims": 20,
    "cache_tables": True,
}

key = jax.random.PRNGKey(0)
lewis_obj = lewis.Lewis45(**default_params)
unifs = jax.random.uniform(key=key, shape=lewis_obj.unifs_shape())
p = jnp.array([0.25, 0.5, 0.75])
berns = unifs < p[None]
berns_order = jnp.arange(0, berns.shape[0])


def test_stage_1():
    # actual
    (
        early_exit_futility,
        data,
        non_dropped_idx,
        pps,
        berns_start,
    ) = lewis_obj.stage_1(berns, berns_order)

    # expected
    early_exit_futility_expected = False
    data_expected = jnp.array([[0, 3], [0, 1], [2, 3]], dtype=int)
    non_dropped_idx_expected = jnp.array([False, True])
    _, pps_expected = lewis_obj.get_pr_best_pps_1__(data_expected)
    berns_start_expected = 3

    # test
    assert jnp.array_equal(early_exit_futility, early_exit_futility_expected)
    assert jnp.array_equal(data, data_expected)
    assert jnp.array_equal(non_dropped_idx, non_dropped_idx_expected)
    assert jnp.array_equal(pps, pps_expected)
    assert jnp.array_equal(berns_start, berns_start_expected)


def test_stage_2():
    # expected stage 1
    data = jnp.array([[1, 3], [0, 1], [2, 3]], dtype=int)
    best_arm = 2
    berns_start = 3

    # actual stage 2
    rej, _ = lewis_obj.stage_2(data, best_arm, berns, berns_order, berns_start)

    # test
    assert jnp.array_equal(rej, False)


def test_inter_stage():
    null_truths = jnp.zeros(default_params["n_arms"] - 1, dtype=bool)
    rej, _ = lewis_obj.simulate(p, null_truths, unifs, berns_order)
    assert jnp.array_equal(rej, False)
