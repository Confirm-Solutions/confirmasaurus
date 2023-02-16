import asyncio

import confirm.adagrid.adagrid as adagrid
import imprint as ip
from confirm.adagrid.validate import ada_validate
from confirm.adagrid.validate import AdaValidate
from confirm.cloud.clickhouse import get_redis_client
from confirm.cloud.redis_heartbeat import HeartbeatThread
from imprint.models.ztest import ZTest1D


def test_heartbeat():
    async def _test():
        redis_con = get_redis_client()
        async with HeartbeatThread(
            redis_con, "test_heartbeat", 2, heartbeat_sleep=0.01
        ) as h:
            assert h.extend_count == 0
            await asyncio.sleep(0)
            assert redis_con.get("test_heartbeat:heartbeat:2") is not None
            assert redis_con.sismember("test_heartbeat:workers", 2)
        assert not redis_con.sismember("test_heartbeat:workers", 2)
        assert h.extend_count > 0

    asyncio.run(_test())


def test_process():
    pass


def test_steal():
    pass


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


def test_init():
    # some database functions should select only this worker's tiles:
    # - bootstrap_lamss
    # - worst_tile
    kwargs = get_test_defaults(ada_validate)
    ada = adagrid.Adagrid()

    async def _test():
        await ada.init(AdaValidate, wait_for_db=True, **kwargs)
        db = ada.db
        tiles_df = db.get_tiles()
        assert db.store.get("config").shape[0] == 1
        assert ada.worker_id == 2
        assert tiles_df.shape[0] == 5
        assert (tiles_df["step_id"] == 0).all()
        assert tiles_df["packet_id"].value_counts().to_dict() == {0: 2, 1: 2, 2: 1}

    asyncio.run(_test())


def test_process_step():
    kwargs = get_test_defaults(ada_validate)
    ada = adagrid.Adagrid()

    async def _test():
        await ada.init(AdaValidate, wait_for_db=True, **kwargs)
        await ada.process_step(0, 0)
        results_df = ada.db.get_results()
        assert results_df.shape[0] == 5

        await ada.process_step(0, 1)
        report = ada.db.get_reports().iloc[-1]
        assert report["status"] == "WORK_DONE"

    asyncio.run(_test())


def test_new_step():
    kwargs = get_test_defaults(ada_validate)
    ada = adagrid.Adagrid()

    async def _test():
        await ada.init(AdaValidate, wait_for_db=True, **kwargs)
        await ada.process_step(0, 0)
        await ada.new_step(0, 1)
        reports = ada.db.get_reports()
        assert reports.shape[0] == 5
        new_step_rpt = reports.iloc[-1]
        assert new_step_rpt["n_refine"] == 3
        assert new_step_rpt["status"] == "NEW_STEP"

    asyncio.run(_test())


def test_coordination():
    pass
