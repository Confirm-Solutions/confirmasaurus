import asyncio

from confirm.cloud.clickhouse import get_redis_client
from confirm.cloud.redis_heartbeat import HeartbeatThread


def test_heartbeat():
    async def _test():
        redis_con = get_redis_client()
        async with HeartbeatThread(
            redis_con, "test_heartbeat", 2, heartbeat_sleep=0.01
        ) as h:
            await asyncio.sleep(0.2)
            refresh_count = h.extend_count
        assert refresh_count > 0

    asyncio.run(_test())


def test_steal():
    pass
