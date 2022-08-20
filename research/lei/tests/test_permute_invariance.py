import jax
import jax.numpy as jnp
from lewis import lewis


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
    "n_pr_sims": 100,
    "n_sig2_sim": 20,
}


def test_posterior_difference_permute():
    lewis_obj = lewis.Lewis45(**default_params)
    n = default_params["n_stage_1"]

    data_1 = jnp.array(
        [
            [0, n],
            [1, n],
            [2, n],
        ]
    )
    out_1 = lewis_obj.posterior_difference(data_1)

    data_2 = jnp.array(
        [
            [0, n],
            [2, n],
            [1, n],
        ]
    )
    out_2 = lewis_obj.posterior_difference(data_2)

    permute = jnp.array([1, 0])

    assert jnp.allclose(out_1, out_2[permute])


def test_pr_best_permute():
    lewis_obj = lewis.Lewis45(**default_params)
    key = jax.random.PRNGKey(10)

    thetas_1 = jax.random.normal(key=key, shape=(100, 3))
    out_1 = lewis_obj.pr_best(thetas_1)

    permute = jnp.array([1, 0, 2])

    thetas_2 = thetas_1[:, permute]
    out_2 = lewis_obj.pr_best(thetas_2)

    assert jnp.allclose(out_1, out_2[permute])


def test_pps_permute():
    lewis_obj = lewis.Lewis45(**default_params)
    lewis_obj.cache_posterior_difference_table(batch_size=int(2**16))

    n = default_params["n_stage_1"]
    key = jax.random.PRNGKey(10)

    data_1 = jnp.array([[0, n], [1, n], [2, n]])
    thetas_1 = jax.random.normal(key=key, shape=(100, 3))
    _, key = jax.random.split(key)
    unifs_1 = jax.random.uniform(key=key, shape=(100, 10, 3))
    out_1 = lewis_obj.pps(data_1, thetas_1, unifs_1)

    permute = jnp.array([0, 2, 1])

    data_2 = data_1[permute]
    thetas_2 = thetas_1[:, permute]
    unifs_2 = unifs_1[..., permute]
    out_2 = lewis_obj.pps(data_2, thetas_2, unifs_2)

    assert jnp.allclose(out_1, out_2[permute[1:] - 1])
