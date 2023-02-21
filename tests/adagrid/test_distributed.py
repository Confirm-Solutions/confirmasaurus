import asyncio
import copy

import numpy as np

import confirm.cloud.clickhouse as ch
import imprint as ip
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
        algo, zone_info = await init(AdaValidate, 1, kwargs)
        assert len(zone_info) == 1
        assert zone_info[0] == [(0, 0), (0, 1), (0, 2)]

        assert algo.db is kwargs["db"]

        for k in kwargs:
            if k in algo.cfg:
                assert algo.cfg[k] == kwargs[k]

        tiles_df = algo.db.get_tiles()
        assert algo.db.store.get("config").shape[0] == 1
        assert algo.cfg["worker_id"] == 2
        assert tiles_df.shape[0] == 5
        assert (tiles_df["step_id"] == 0).all()
        assert tiles_df["packet_id"].value_counts().to_dict() == {0: 2, 1: 2, 2: 1}

    asyncio.run(_test())


def test_init_join(ch_db):
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = ch_db

    async def _test():
        algo1, zone_info1 = await init(AdaValidate, 1, kwargs)

        kwargs2 = copy.copy(kwargs)
        kwargs2["g"] = None
        kwargs2["lam"] = -4
        kwargs2["overrides"] = dict(packet_size=3)
        algo, zone_info2 = await init(AdaValidate, 1, kwargs2)
        assert zone_info2 is None
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

    async def _test():
        algo, zone_info = await init(AdaValidate, 1, kwargs)
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


def test_new_step():
    kwargs = get_test_defaults(ada_validate)

    async def _test():
        algo, _ = await init(AdaValidate, 1, kwargs)
        await process_packet_set(algo, 0, [(0, i) for i in range(3)])

        status, n_packets, report_task = await new_step(algo, 0, 1)
        assert status == WorkerStatus.NEW_STEP
        assert n_packets == 3
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
        assert (done["refine"] == 1).all()
        assert (done["deepen"] == 0).all()
        assert (done["active"] == 0).all()
        assert (done["step_id"] == 0).all()

        await report_task
        report = algo.db.get_reports().iloc[-1]
        assert report["status"] == "NEW_STEP"
        assert report["n_refine"] == 3

    asyncio.run(_test())


def test_reload_zone_info():
    kwargs = get_test_defaults(ada_validate)

    async def _test():
        algo, _ = await init(AdaValidate, 1, kwargs)
        await process_packet_set(algo, 0, [(0, i) for i in range(3)])
        _, n_packets, _ = await new_step(algo, 0, 1)

        kwargs2 = kwargs.copy()
        kwargs2["db"] = algo.db
        kwargs2["g"] = None
        algo, zone_info = await init(AdaValidate, 1, kwargs2)
        assert zone_info[0] == dict(step_id=1, n_packet=n_packets)

    asyncio.run(_test())


def test_coordinate(ch_db):
    kwargs = get_test_defaults(ada_validate)
    kwargs["db"] = ch_db

    async def _test():
        algo, _ = await init(AdaValidate, 1, kwargs)
        await process_packet_set(algo, 0, [(0, i) for i in range(3)])
        pre_df = ch_db.get_results()
        assert pre_df["zone_id"].unique() == [0]
        assert (pre_df["coordination_id"] == 0).all()

        status, lazy_tasks = await coordinate(algo, 0, 2)
        assert status == WorkerStatus.COORDINATED
        post_df = ch_db.get_results()
        assert (np.sort(post_df["zone_id"].unique()) == [0, 1]).all()
        assert (post_df["coordination_id"] == 1).all()
        assert post_df.shape[0] == pre_df.shape[0]

        await asyncio.gather(*lazy_tasks)
        zone_map_df = ch._query_df(ch_db.client, "select * from zone_mapping")
        assert (zone_map_df["old_zone_id"] == 0).all()
        assert (zone_map_df["id"].isin(pre_df["id"])).all()
        assert (zone_map_df["id"].isin(post_df["id"])).all()
        assert (
            zone_map_df.set_index("id").loc[post_df["id"], "new_zone_id"].values
            == post_df["zone_id"]
        ).all()

    asyncio.run(_test())


# def test_new_step():
#     kwargs = get_test_defaults(ada_validate)
#     ada = adagrid.Adagrid()

#     async def _test():
#         await ada.init(AdaValidate, wait_for_db=True, **kwargs)
#         await ada.process_step(0, 0)
#         await ada.new_step(0, 1)
#         reports = ada.db.get_reports()
#         assert reports.shape[0] == 5
#         new_step_rpt = reports.iloc[-1]
#         assert new_step_rpt["n_refine"] == 3
#         assert new_step_rpt["status"] == "NEW_STEP"

#     asyncio.run(_test())
