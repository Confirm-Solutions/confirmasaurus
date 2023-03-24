import jax
import numpy as np

import confirm.models.wd41 as wd41


def test_wd41_zstats():
    model = wd41.WD41(0, 10000, ignore_intersection=True)
    sim_vmap = jax.vmap(model.sim, in_axes=(0, None, None, None, None, None))
    p1 = (0.3, 0.3, 0.3, 0.3)
    # p1 = (0.386393, 0.169736, 0.351489, 0.625392)
    stats = sim_vmap(model.unifs, *p1, True)
    stat_names = [
        "hyptnbc_zstat",
        "hypfull_zstat",
        "ztnbc_stage1",
        "ztnbc_stage2",
        "zfull_stage1",
        "zfull_stage2",
    ]
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 9))
    # for i, S in enumerate(stat_names):
    #     plt.subplot(2, 3, i + 1)
    #     plt.title(S)
    #     plt.hist(stats[S][~np.isinf(stats[S])], bins=25)
    # plt.show()
    for S in stat_names:
        vals = stats[S][~np.isinf(stats[S])]
        vneg = np.percentile(vals, 15.865)
        vpos = np.percentile(vals, 84.135)
        # should be approximately 2 sigma
        assert np.abs((vpos - vneg) - 2) < 0.07


def test_wd41(snapshot):
    model = wd41.WD41(0, 100, ignore_intersection=True)
    theta = np.array([[0, 0, 0, 0]], dtype=np.float32)
    stats = model.sim_batch(
        0,
        1000,
        theta,
        np.array([[True, True]], dtype=np.float32),
    )[0]
    np.testing.assert_allclose(stats, snapshot(stats), rtol=1e-6, atol=1e-6)
