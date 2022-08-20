import numpy as np
from lewis import lewis


default_params = {
    "n_arms": 3,
    "n_stage_1": 10,
    "n_stage_2": 10,
    "n_stage_1_interims": 3,
    "n_stage_1_add_per_interim": 4,
    "n_stage_2_add_per_interim": 4,
    "futility_threshold": 0.1,
    "pps_threshold_lower": 0.1,
    "pps_threshold_upper": 0.9,
    "posterior_difference_threshold": 0.05,
    "rejection_threshold": 0.05,
    "n_pr_sims": 100,
    "n_sig2_sim": 20,
    "batch_size": 2**10,
}


def run_n_configs_test(actual, expected):
    def check_equality(actual, expected):
        # lexicographical sort to order the rows consistently
        actual_sorted = np.lexsort(actual)
        expected_sorted = np.lexsort(expected)
        assert np.array_equal(actual_sorted, expected_sorted)

    for n_configs, n_configs_expected in zip(actual, expected):
        check_equality(n_configs, n_configs_expected)


def make_expected(
    n_configs_pr_best_pps_1_expected,
    n_stage_2,
    n_stage_2_add_per_interim,
):
    n_configs_pps_2_expected = n_configs_pr_best_pps_1_expected
    n_configs_pps_2_expected[:, -2:] += n_stage_2
    n_configs_pd_expected = n_configs_pps_2_expected
    n_configs_pd_expected[:, -2:] += n_stage_2_add_per_interim
    expected = (
        n_configs_pr_best_pps_1_expected,
        n_configs_pps_2_expected,
        n_configs_pd_expected,
    )
    return expected


def test_3_arms_0_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 3
    default_params["n_stage_1"] = 10
    default_params["n_stage_2"] = 10
    default_params["n_stage_1_interims"] = 0
    default_params["n_stage_1_add_per_interim"] = 4
    default_params["n_stage_2_add_per_interim"] = 4

    lewis_obj = lewis.Lewis45(**default_params)
    actual = lewis_obj.make_n_configs__()

    # expected values
    n_configs_pr_best_pps_1_expected = np.array(
        [
            [10, 10, 10],
        ]
    )
    expected = make_expected(
        n_configs_pr_best_pps_1_expected,
        default_params["n_stage_2"],
        default_params["n_stage_2_add_per_interim"],
    )

    # tests
    run_n_configs_test(actual, expected)


def test_3_arms_1_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 3
    default_params["n_stage_1"] = 10
    default_params["n_stage_2"] = 10
    default_params["n_stage_1_interims"] = 1
    default_params["n_stage_1_add_per_interim"] = 4
    default_params["n_stage_2_add_per_interim"] = 4

    lewis_obj = lewis.Lewis45(**default_params)
    actual = lewis_obj.make_n_configs__()

    # expected values
    n_configs_pr_best_pps_1_expected = np.array(
        [
            [10, 10, 10],
            [11, 11, 11],
            [10, 12, 12],
        ]
    )
    expected = make_expected(
        n_configs_pr_best_pps_1_expected,
        default_params["n_stage_2"],
        default_params["n_stage_2_add_per_interim"],
    )

    # tests
    run_n_configs_test(actual, expected)


def test_3_arms_2_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 3
    default_params["n_stage_1"] = 5
    default_params["n_stage_2"] = 15
    default_params["n_stage_1_interims"] = 2
    default_params["n_stage_1_add_per_interim"] = 7
    default_params["n_stage_2_add_per_interim"] = 4

    lewis_obj = lewis.Lewis45(**default_params)
    actual = lewis_obj.make_n_configs__()

    # expected values
    n_configs_pr_best_pps_1_expected = np.array(
        [
            [5, 5, 5],
            [7, 7, 7],
            [5, 8, 8],
            [9, 9, 9],
            [7, 10, 10],
            [5, 11, 11],
        ]
    )
    expected = make_expected(
        n_configs_pr_best_pps_1_expected,
        default_params["n_stage_2"],
        default_params["n_stage_2_add_per_interim"],
    )

    # tests
    run_n_configs_test(actual, expected)


def test_4_arms_0_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 4
    default_params["n_stage_1"] = 10
    default_params["n_stage_2"] = 10
    default_params["n_stage_1_interims"] = 0
    default_params["n_stage_1_add_per_interim"] = 4
    default_params["n_stage_1_add_per_interim"] = 10

    lewis_obj = lewis.Lewis45(**default_params)
    actual = lewis_obj.make_n_configs__()

    # expected values
    n_configs_pr_best_pps_1_expected = np.array(
        [
            [10, 10, 10, 10],
        ]
    )
    expected = make_expected(
        n_configs_pr_best_pps_1_expected,
        default_params["n_stage_2"],
        default_params["n_stage_2_add_per_interim"],
    )

    # tests
    run_n_configs_test(actual, expected)


def test_4_arms_1_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 4
    default_params["n_stage_1"] = 10
    default_params["n_stage_2"] = 10
    default_params["n_stage_1_interims"] = 1
    default_params["n_stage_1_add_per_interim"] = 4
    default_params["n_stage_2_add_per_interim"] = 20

    lewis_obj = lewis.Lewis45(**default_params)
    actual = lewis_obj.make_n_configs__()

    # expected values
    n_configs_pr_best_pps_1_expected = np.array(
        [
            [10, 10, 10, 10],
            [11, 11, 11, 11],
            [10, 11, 11, 11],
            [10, 10, 12, 12],
        ]
    )
    expected = make_expected(
        n_configs_pr_best_pps_1_expected,
        default_params["n_stage_2"],
        default_params["n_stage_2_add_per_interim"],
    )

    # tests
    run_n_configs_test(actual, expected)


def test_4_arms_2_interim():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 4
    default_params["n_stage_1"] = 10
    default_params["n_stage_2"] = 10
    default_params["n_stage_1_interims"] = 2
    default_params["n_stage_1_add_per_interim"] = 4
    default_params["n_stage_2_add_per_interim"] = 1

    lewis_obj = lewis.Lewis45(**default_params)
    actual = lewis_obj.make_n_configs__()

    # expected values
    n_configs_pr_best_pps_1_expected = np.array(
        [
            [10, 10, 10, 10],
            [11, 11, 11, 11],
            [10, 11, 11, 11],
            [10, 10, 12, 12],
            [12, 12, 12, 12],
            [11, 12, 12, 12],
            [11, 11, 13, 13],
            [10, 12, 12, 12],
            [10, 11, 13, 13],
            [10, 10, 14, 14],
        ]
    )
    expected = make_expected(
        n_configs_pr_best_pps_1_expected,
        default_params["n_stage_2"],
        default_params["n_stage_2_add_per_interim"],
    )

    # tests
    run_n_configs_test(actual, expected)


def test_3_arms_2_interms_settings():
    # re-setting parameters to make it clear which parameters affect this function.
    default_params["n_arms"] = 3
    default_params["n_stage_1"] = 5
    default_params["n_stage_2"] = 15
    default_params["n_stage_1_interims"] = 2
    default_params["n_stage_1_add_per_interim"] = 7
    default_params["n_stage_2_add_per_interim"] = 4

    lewis_obj = lewis.Lewis45(**default_params)
    (
        n_configs_max_mask,
        _,
        hashes_pr_best_pps_1,
        _,
        _,
        hashes_pps_2,
        _,
        _,
        hashes_pd,
        _,
    ) = lewis_obj.n_configs_setting__()

    # expected values
    n_configs_max_mask_expected = np.array([1, 30, 30**2])

    # tests
    assert np.array_equal(n_configs_max_mask, n_configs_max_mask_expected)
    assert np.unique(hashes_pr_best_pps_1).shape == hashes_pr_best_pps_1.shape
    assert np.unique(hashes_pps_2).shape == hashes_pps_2.shape
    assert np.unique(hashes_pd).shape == hashes_pd.shape
