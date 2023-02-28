import asyncio
import copy

import numpy as np
import pandas as pd

import imprint as ip
from confirm.adagrid.backend import LocalBackend
from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.coordinate import coordinate
from confirm.adagrid.init import init
from confirm.adagrid.step import new_step
from confirm.adagrid.step import process_packet
from confirm.adagrid.step import process_packet_set
from confirm.adagrid.validate import ada_validate
from confirm.adagrid.validate import AdaValidate
from imprint.models.ztest import ZTest1D


def get_test_defaults(f):
    import inspect

    sig = inspect.signature(f)
    kwargs = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    kwargs["prod"] = False
    kwargs["verify"] = True
    kwargs["model_type"] = ZTest1D
    kwargs["g"] = ip.cartesian_grid(
        theta_min=[-1], theta_max=[1], n=[10], null_hypos=[ip.hypo("x0 < 0")]
    )
    kwargs["lam"] = -1.96
    kwargs["packet_size"] = 2
    return kwargs


def test_init_first(both_dbs):
    # TODO: some database functions should select only this zone's tiles:
    # - bootstrap_lamss
    # - worst_tile
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = both_dbs

    async def _test():
        algo, incomplete_packets, zone_info = await init(
            AdaValidate, True, 1, 1, kwargs
        )
        assert incomplete_packets == [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
        assert len(zone_info) == 1
        assert zone_info[0] == 0

        assert algo.db is kwargs["db"]

        for k in kwargs:
            if k in algo.cfg:
                assert algo.cfg[k] == kwargs[k]

        tiles_df = algo.db.get_tiles()
        assert algo.db.store.get("config").shape[0] == 1
        assert algo.cfg["worker_id"] == 1
        assert tiles_df.shape[0] == 5
        assert (tiles_df["step_id"] == 0).all()
        assert tiles_df["packet_id"].value_counts().to_dict() == {0: 2, 1: 2, 2: 1}

    asyncio.run(_test())


def test_init_join(both_dbs):
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = both_dbs

    async def _test():
        algo1, _, _ = await init(AdaValidate, True, 1, 1, kwargs)

        kwargs2 = copy.copy(kwargs)
        kwargs2["g"] = None
        kwargs2["lam"] = -4
        kwargs2["overrides"] = dict(packet_size=3)
        algo, incomplete2, zone_info2 = await init(AdaValidate, True, 1, 1, kwargs2)
        assert incomplete2 == [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
        assert len(zone_info2) == 1
        assert zone_info2[0] == 0

        assert algo.db is kwargs["db"]
        assert algo.cfg["packet_size"] == 3
        for k in algo1.cfg:
            if k not in kwargs:
                continue
            if k == "packet_size":
                continue
            assert algo.cfg[k] == algo1.cfg[k]

    asyncio.run(_test())


def test_process(both_dbs):
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = both_dbs

    async def _test():
        algo, incomplete, zone_info = await init(AdaValidate, True, 1, 1, kwargs)
        await asyncio.gather(*await process_packet(algo, 0, 0, 0))
        results_df = algo.db.get_results()
        assert results_df.shape[0] == 2
        assert (results_df["packet_id"] == 0).all()

        # Check that process is idempotent
        await asyncio.gather(*await process_packet(algo, 0, 0, 0))
        results_df = algo.db.get_results()
        assert results_df.shape[0] == 2
        assert (results_df["packet_id"] == 0).all()

        await asyncio.gather(*await process_packet(algo, 0, 0, 1))
        await asyncio.gather(*await process_packet(algo, 0, 0, 2))
        results_df = algo.db.get_results()
        assert results_df.shape[0] == 5

        report = algo.db.get_reports().iloc[-1]
        assert report["status"] == "WORKING"

    asyncio.run(_test())


def test_new_step(both_dbs):
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = both_dbs

    async def _test():
        algo, _, _ = await init(AdaValidate, True, 1, 1, kwargs)
        await process_packet_set(algo, [(0, 0, i) for i in range(3)])

        status, n_packets, report_task = await new_step(algo, 0, 1)
        # call new_step twice to confirm idempotency
        status2, n_packets2, report_task2 = await new_step(algo, 0, 1)
        assert status == WorkerStatus.NEW_STEP
        assert status2 == WorkerStatus.ALREADY_EXISTS

        assert n_packets == 3
        assert n_packets2 == 3

        tiles_df = algo.db.get_tiles()
        results_df = algo.db.get_results()

        assert tiles_df.shape[0] == 11
        assert results_df.shape[0] == 5

        new_tiles = tiles_df.iloc[5:]
        assert (new_tiles["zone_id"] == 0).all()
        assert (new_tiles["step_id"] == 1).all()
        assert (new_tiles["creator_id"] == algo.cfg["worker_id"]).all()

        done = algo.db.get_done()[1:]
        assert done.shape[0] == 3
        assert (done["refine"] > 0).all()
        assert (done["deepen"] == 0).all()
        assert (done["active"] == 0).all()
        assert (done["step_id"] == 0).all()

        report = await report_task
        assert report["status"] == "NEW_STEP"
        assert report["n_refine"] == 3

    asyncio.run(_test())


def test_reload_zone_info(both_dbs):
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = both_dbs

    async def _test():
        algo, incomplete, _ = await init(AdaValidate, True, 1, 1, kwargs)
        await process_packet_set(algo, incomplete)
        _, n_packets, _ = await new_step(algo, 0, 1)

        kwargs2 = kwargs.copy()
        kwargs2["db"] = algo.db
        kwargs2["g"] = None
        algo, incomplete, zone_info = await init(AdaValidate, True, 1, 1, kwargs2)
        assert incomplete == [(0, 1, 0), (0, 1, 1), (0, 1, 2)]
        assert zone_info[0] == 1

    asyncio.run(_test())


def test_coordinate(both_dbs):
    db = both_dbs
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = db

    async def _test():
        algo, incomplete, _ = await init(AdaValidate, True, 1, 1, kwargs)
        await process_packet_set(algo, incomplete)
        pre_df = db.get_results()
        assert pre_df["zone_id"].unique() == [0]
        assert (pre_df["coordination_id"] == 0).all()

        status, lazy_tasks, zone_steps = await coordinate(algo, 0, 2)
        assert zone_steps == {0: 0, 1: 0}
        assert status == WorkerStatus.COORDINATED
        post_df = db.get_results()
        assert (np.sort(post_df["zone_id"].unique()) == [0, 1]).all()
        assert (post_df["coordination_id"] == 1).all()
        assert post_df.shape[0] == pre_df.shape[0]

        await asyncio.gather(*lazy_tasks)
        zone_map_df = db.get_zone_mapping()
        assert (zone_map_df["coordination_id"] == 1).all()
        assert (zone_map_df["old_zone_id"] == 0).all()
        assert (zone_map_df["id"].isin(pre_df["id"])).all()
        assert (zone_map_df["id"].isin(post_df["id"])).all()
        assert (
            zone_map_df.set_index("id").loc[post_df["id"], "zone_id"].values
            == post_df["zone_id"]
        ).all()

        # Coordinate again to check idempotency
        status2, _, _ = await coordinate(algo, 0, 2)
        assert status == status2
        idem_df = db.get_results()
        pd.testing.assert_frame_equal(
            idem_df.drop("coordination_id", axis=1),
            post_df.drop("coordination_id", axis=1),
        )
        assert (idem_df["coordination_id"] == 2).all()

    asyncio.run(_test())


def test_idempotency(both_dbs):
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = both_dbs
    backend = LocalBackend(n_zones=1)
    backend.run(AdaValidate, kwargs)
    reports = both_dbs.get_reports()
    backend.run(AdaValidate, kwargs)
    backend.run(AdaValidate, kwargs)
    reports2 = both_dbs.get_reports()
    assert reports.shape[0] + 2 == reports2.shape[0]
    drop_cols = [c for c in reports2.columns if "runtime" in c]
    pd.testing.assert_series_equal(
        reports2.iloc[-3].drop(drop_cols),
        reports2.iloc[-2].drop(drop_cols),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        reports2.iloc[-2].drop(drop_cols),
        reports2.iloc[-1].drop(drop_cols),
        check_names=False,
    )
