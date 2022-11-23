from unittest import mock

import numpy as np
import pandas as pd
import pytest

import confirm.imprint as ip
from confirm.models.ztest import ZTest1D


@pytest.mark.slow
@mock.patch("time.time", mock.MagicMock(return_value=100))
def test_adagrid(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, ada = ip.ada_tune(ZTest1D, g, nB=5, db_type=ip.db.DuckDBTiles)
    lamss = reports[-1]["lamss"]
    np.testing.assert_allclose(lamss, snapshot(lamss))
    assert iter == snapshot(iter)

    all_tiles_df = ada.tiledb.get_all()
    pd.testing.assert_frame_equal(
        all_tiles_df, snapshot(all_tiles_df), check_dtype=False
    )

    # Compare DuckDB against pandas
    _, _, ada2 = ip.ada_tune(ZTest1D, g, nB=5, db_type=ip.db.PandasTiles)
    pd.testing.assert_frame_equal(
        ada.tiledb.get_all().drop(["id", "parent_id"], axis=1),
        ada2.tiledb.get_all().drop(["id", "parent_id"], axis=1),
    )
