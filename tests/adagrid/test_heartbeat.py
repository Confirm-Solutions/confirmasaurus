import asyncio

from adagrid.heartbeat import HeartbeatThread


# testing async https://stackoverflow.com/a/70016047/3817027
async def test_heartbeat():
    async with HeartbeatThread(2, heartbeat_sleep=0.01) as h:
        await asyncio.sleep(0.2)
        refresh_count = h.extend_count
    assert refresh_count > 0
