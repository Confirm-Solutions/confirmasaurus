import asyncio
import copy

import numpy as np
import pandas as pd

import imprint as ip
from confirm.adagrid.backend import entrypoint
from confirm.adagrid.backend import LocalBackend
from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.coordinate import coordinate
from confirm.adagrid.db import DuckDBTiles
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


def test_init_first():
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = DuckDBTiles.connect()

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
                if k == "model_kwargs":
                    assert algo.cfg[k] == {}
                else:
                    assert algo.cfg[k] == kwargs[k]

        tiles_df = algo.db.get_tiles()
        assert algo.db.get_config().shape[0] == 1
        assert algo.cfg["worker_id"] == 1
        assert tiles_df.shape[0] == 5
        assert (tiles_df["step_id"] == 0).all()
        assert tiles_df["packet_id"].value_counts().to_dict() == {0: 2, 1: 2, 2: 1}

    asyncio.run(_test())


def test_init_join():
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = DuckDBTiles.connect()

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


def test_process():
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = DuckDBTiles.connect()
    backend = LocalBackend()

    async def _test():
        algo, incomplete, zone_info = await init(AdaValidate, True, 1, 1, kwargs)
        async with backend.setup(AdaValidate, algo, kwargs):
            await asyncio.gather(*await process_packet(backend, algo, 0, 0, 0))
            results_df = algo.db.get_results()
            assert results_df.shape[0] == 2
            assert (results_df["packet_id"] == 0).all()

            # Check that process is idempotent
            await asyncio.gather(*await process_packet(backend, algo, 0, 0, 0))
            results_df = algo.db.get_results()
            assert results_df.shape[0] == 2
            assert (results_df["packet_id"] == 0).all()

            await asyncio.gather(*await process_packet(backend, algo, 0, 0, 1))
            await asyncio.gather(*await process_packet(backend, algo, 0, 0, 2))
        results_df = algo.db.get_results()
        assert results_df.shape[0] == 5

        report = algo.db.get_reports().iloc[-1]
        assert report["status"] == "WORKING"

    asyncio.run(_test())


def test_new_step():
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = DuckDBTiles.connect()
    backend = LocalBackend()

    async def _test():
        algo, _, _ = await init(AdaValidate, True, 1, 1, kwargs)
        async with backend.setup(AdaValidate, algo, kwargs):
            for i in range(3):
                await process_packet_set(backend, algo, [(0, 0, i) for i in range(3)])

        status, tiles_df, before_tasks, report_task = await new_step(algo, 0, 1)
        await asyncio.gather(*before_tasks)

        # call new_step twice to confirm idempotency
        status2, tiles_df2, before_tasks, report_task2 = await new_step(algo, 0, 1)
        assert status == WorkerStatus.NEW_STEP
        assert status2 == WorkerStatus.ALREADY_EXISTS

        assert tiles_df["packet_id"].nunique() == 3
        assert tiles_df2["packet_id"].nunique() == 3

        all_tiles_df = algo.db.get_tiles()
        results_df = algo.db.get_results()

        assert all_tiles_df.shape[0] == 11
        assert results_df.shape[0] == 5

        new_tiles = tiles_df[tiles_df["step_id"] == 1]
        assert new_tiles.shape[0] == 6
        assert (new_tiles["zone_id"] == 0).all()
        assert (new_tiles["creator_id"] == algo.cfg["worker_id"]).all()

        done = algo.db.get_done().sort_values(by=["id"])[1:]
        assert done.shape[0] == 3
        assert (done["refine"] > 0).all()
        assert (done["deepen"] == 0).all()
        assert (done["active"] == 0).all()
        assert (done["step_id"] == 0).all()

        report = await report_task[2]
        assert report["status"] == "NEW_STEP"
        assert report["n_refine"] == 3

    asyncio.run(_test())


def test_reload_zone_info():
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = DuckDBTiles.connect()
    backend = LocalBackend()

    async def _test():
        algo, incomplete, _ = await init(AdaValidate, True, 1, 1, kwargs)
        async with backend.setup(AdaValidate, algo, kwargs):
            await process_packet_set(backend, algo, incomplete)
        _, _, before_tasks, _ = await new_step(algo, 0, 1)
        await asyncio.gather(*before_tasks)

        kwargs2 = kwargs.copy()
        kwargs2["db"] = algo.db
        kwargs2["g"] = None
        algo, incomplete, zone_info = await init(AdaValidate, True, 1, 1, kwargs2)
        assert incomplete == [(0, 1, 0), (0, 1, 1), (0, 1, 2)]
        assert zone_info[0] == 1

    asyncio.run(_test())


def test_coordinate():
    kwargs = get_test_defaults(ada_validate)
    db = DuckDBTiles.connect()
    kwargs["db"] = db
    backend = LocalBackend()

    async def _test():
        algo, incomplete, _ = await init(AdaValidate, True, 1, 1, kwargs)
        async with backend.setup(AdaValidate, algo, kwargs):
            await process_packet_set(backend, algo, incomplete)
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


def test_idempotency():
    kwargs = get_test_defaults(ada_validate)
    db = entrypoint(AdaValidate, kwargs)
    reports = db.get_reports()

    del kwargs["g"]
    kwargs["db"] = db
    entrypoint(AdaValidate, kwargs)
    reports2 = db.get_reports()
    assert reports.shape[0] + 1 == reports2.shape[0]
    drop_cols = [c for c in reports2.columns if "runtime" in c]
    pd.testing.assert_series_equal(
        reports2.iloc[-2].drop(drop_cols),
        reports2.iloc[-1].drop(drop_cols),
        check_names=False,
    )
