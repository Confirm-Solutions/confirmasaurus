import jax.numpy as jnp

from confirm.lewislib import lewis


default_params = {
    "n_arms": 3,
    "n_stage_1": 3,
    "n_stage_2": 3,
    "n_stage_1_interims": 1,
    "n_stage_1_add_per_interim": 10,
    "n_stage_2_add_per_interim": 4,
    "stage_1_futility_threshold": 0.1,
    "stage_2_futility_threshold": 0.1,
    "stage_1_efficacy_threshold": 0.1,
    "stage_2_efficacy_threshold": 0.9,
    "inter_stage_futility_threshold": 0.8,
    "posterior_difference_threshold": 0.05,
    "rejection_threshold": 0.05,
}


def test_get_posterior_difference():
    lewis_obj = lewis.Lewis45(**default_params)
    lewis_obj.pd_table = lewis_obj._posterior_difference_table(batch_size=int(2**16))
    n = lewis_obj.n_configs_pd[2]
    y = jnp.array([5, 1, 2])
    data = jnp.stack((y, n), axis=-1)
    out_1 = lewis_obj._get_posterior_difference(data)
    permute = jnp.array([0, 2, 1])
    data_2 = data[permute]
    out_2 = lewis_obj._get_posterior_difference(data_2)
    assert jnp.array_equal(out_1, out_2[permute[1:] - 1])
