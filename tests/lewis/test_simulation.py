import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    "batch_size": 2**4,
    "key": jax.random.PRNGKey(1),
    "n_pr_sims": 100,
    "n_sig2_sims": 20,
    # NOTE: because we are caching tables, this code *does not* test the table
    # construction!
    "cache_tables": Path(__file__).resolve().parent.joinpath("lewis.pkl"),
}


def lewis_small():
    key = jax.random.PRNGKey(0)
    lewis_obj = lewis.Lewis45(**default_params)
    unifs = jax.random.uniform(key=key, shape=lewis_obj.unifs_shape())
    p = jnp.array([0.25, 0.5, 0.75])
    berns = unifs < p[None]
    berns_order = jnp.arange(0, berns.shape[0])
    return (lewis_obj, unifs, p, berns, berns_order)


@pytest.fixture(name="lewis_small", scope="session")
def lewis_small_fixture():
    return lewis_small()


def test_save_load(tmp_path):
    params = default_params.copy()
    params["cache_tables"] = False
    L1 = lewis.Lewis45(**default_params)
    path = os.path.join(tmp_path, "tables.pkl")
    if os.path.exists(path):
        os.remove(path)
    L1.save_tables(path)

    params = default_params.copy()
    params["cache_tables"] = path
    L2 = lewis.Lewis45(**params)
    assert L2.loaded_tables
    np.testing.assert_allclose(L1.pd_table.tables[0], L2.pd_table.tables[0])


def test_stage_1(lewis_small):
    lewis_obj, _, _, berns, berns_order = lewis_small
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
    _, pps_expected = lewis_obj._get_pr_best_pps_1(data_expected)
    berns_start_expected = 3

    # test
    assert jnp.array_equal(early_exit_futility, early_exit_futility_expected)
    assert jnp.array_equal(data, data_expected)
    assert jnp.array_equal(non_dropped_idx, non_dropped_idx_expected)
    assert jnp.array_equal(pps, pps_expected)
    assert jnp.array_equal(berns_start, berns_start_expected)


def test_stage_2(lewis_small):
    lewis_obj, _, p, berns, berns_order = lewis_small
    # expected stage 1
    data = jnp.array([[1, 3], [0, 1], [2, 3]], dtype=int)
    best_arm = 2
    berns_start = 3

    # actual stage 2
    test_stat, best_arm, _ = lewis_obj.stage_2(
        data, best_arm, berns, berns_order, berns_start, p
    )
    np.testing.assert_allclose(test_stat, 1.0)
    assert best_arm == 2


def test_inter_stage(lewis_small):
    lewis_obj, unifs, p, _, berns_order = lewis_small
    test_stat, best_arm, _ = lewis_obj.simulate(p, unifs, berns_order)
    np.testing.assert_allclose(test_stat, 2.0)
    assert best_arm == 2
