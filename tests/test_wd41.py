import numpy as np
import pytest

import confirm.models.wd41 as wd41


@pytest.mark.slow
def test_wd41(snapshot):
    model = wd41.WD41(0, 100, ignore_intersection=True)
    stats = model.sim_batch(
        0,
        1000,
        np.array([[0, 0, 0, 0]], dtype=np.float32),
        np.array([[True, True]], dtype=np.float32),
    )[0]
    np.testing.assert_allclose(stats, snapshot(stats))
