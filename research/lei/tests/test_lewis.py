import jax.numpy as jnp
from lewis import lewis


default_params = {
    "n_arms": 3,
    "n_stage_1": 10,
    "n_stage_2": 10,
    "n_interims": 3,
    "n_add_per_interim": 4,
    "futility_threshold": 0.1,
    "pps_threshold_lower": 0.1,
    "pps_threshold_upper": 0.9,
    "posterior_difference_threshold": 0.05,
    "rejection_threshold": 0.05,
    "n_pr_sims": 100,
    "n_sig2_sim": 20,
    "batch_size": 2**10,
}


def test_n_configs_0_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 3
    default_params["n_stage_1"] = 10
    default_params["n_stage_2"] = 10
    default_params["n_interims"] = 0
    default_params["n_add_per_interim"] = 4

    lewis_obj = lewis.Lewis45(**default_params)
    (
        n_configs_ph2,
        n_configs_ph3,
    ) = lewis_obj.make_n_configs__()

    # lexicographical sort to order the rows consistently
    n_configs_ph2 = jnp.lexsort(n_configs_ph2)
    n_configs_ph3 = jnp.lexsort(n_configs_ph3)

    # expected values
    n_configs_ph2_expected = jnp.lexsort(
        jnp.array(
            [
                [10, 10, 10],
            ]
        )
    )
    n_configs_ph3_expected = jnp.lexsort(
        jnp.array(
            [
                [10, 20, 20],
            ]
        )
    )

    # tests
    jnp.array_equal(n_configs_ph2, n_configs_ph2_expected)
    jnp.array_equal(n_configs_ph3, n_configs_ph3_expected)


def test_n_configs_1_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 3
    default_params["n_stage_1"] = 10
    default_params["n_stage_2"] = 10
    default_params["n_interims"] = 1
    default_params["n_add_per_interim"] = 4

    lewis_obj = lewis.Lewis45(**default_params)
    (
        n_configs_ph2,
        n_configs_ph3,
    ) = lewis_obj.make_n_configs__()

    # lexicographical sort to order the rows consistently
    n_configs_ph2 = jnp.lexsort(n_configs_ph2)
    n_configs_ph3 = jnp.lexsort(n_configs_ph3)

    # expected values
    n_configs_ph2_expected = jnp.lexsort(
        jnp.array(
            [
                [10, 10, 10],
                [11, 11, 11],
                [10, 12, 12],
            ]
        )
    )
    n_configs_ph3_expected = jnp.lexsort(
        jnp.array(
            [
                [10, 20, 20],
            ]
        )
    )

    # tests
    jnp.array_equal(n_configs_ph2, n_configs_ph2_expected)
    jnp.array_equal(n_configs_ph3, n_configs_ph3_expected)
