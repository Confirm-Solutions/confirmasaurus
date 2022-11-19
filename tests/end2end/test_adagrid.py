import pandas as pd
import pytest

import confirm.mini_imprint as ip
from confirm.models.ztest import ZTest1D


@pytest.mark.slow
def test_adagrid(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    ada, reports = ip.ada_tune(ZTest1D, g, nB=5, db_type=ip.db.DuckDBTiles)
    snapshot(ada.tiledb.get_all())

    # Compare DuckDB against pandas
    ada2, _ = ip.ada_tune(ZTest1D, g, nB=5, db_type=ip.db.PandasTiles)
    pd.testing.assert_frame_equal(ada.tiledb.get_all(), ada2.tiledb.get_all())
