from unittest import mock

import numpy as np
import pandas as pd
import pytest

import confirm.imprint as ip
from confirm.models.ztest import ZTest1D


@pytest.mark.slow
@mock.patch("confirm.imprint.grid.uuid_timer", mock.MagicMock(return_value=100))
def test_adagrid(snapshot):
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    iter, reports, ada = ip.ada_tune(ZTest1D, g=g, nB=5, tile_batch_size=1)
    lamss = reports[-1]["lamss"]
    np.testing.assert_allclose(lamss, snapshot(lamss))
    assert iter == snapshot(iter)

    all_tiles_df = ada.db.get_all()
    pd.testing.assert_frame_equal(
        all_tiles_df, snapshot(all_tiles_df), check_dtype=False
    )

    # Compare DuckDB against pandas
    pd_db = ip.db.PandasDB()
    _, _, ada2 = ip.ada_tune(ZTest1D, g=g, db=pd_db, nB=5, tile_batch_size=1)
    pd.testing.assert_frame_equal(
        ada.db.get_all().drop(["id", "parent_id"], axis=1),
        ada2.db.get_all().drop(["id", "parent_id"], axis=1),
    )


@pytest.mark.slow
@mock.patch("confirm.imprint.grid.uuid_timer", mock.MagicMock(return_value=100))
def test_adagrid_clickhouse(snapshot):
    import confirm.cloud.clickhouse as ch

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])

    db = ch.Clickhouse.connect()
    iter, reports, ada = ip.ada_tune(ZTest1D, g=g, db=db, nB=5, tile_batch_size=1)
    lamss = reports[-1]["lamss"]
    np.testing.assert_allclose(lamss, snapshot(lamss))
    db.client.command(f"drop database {db.job_id}")


# def test_adagrid_checkpointing():
#     g = ip.cartesian_grid(
#           theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
#     db = ip.db.DuckDB.connect()
#     ada_intermediate = ip.ada_tune(ZTest1D, g=g, db=db, nB=3, init_K=4, n_iter=2)
#     ada_final = ip.ada_tune(ZTest1D, db=db, nB=3, init_K=4, n_iter=1)


def main():
    import confirm.cloud.clickhouse as ch

    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    db = ch.Clickhouse.connect()
    # db = ch.Clickhouse.connect(
    #     host='localhost', port='8123', username='default', password='')
    iter, reports, ada = ip.ada_tune(
        ZTest1D, db=db, g=g, nB=5, tile_batch_size=1, n_iter=8
    )


if __name__ == "__main__":
    main()
