import asyncio
import time

import redis

import imprint.log

logger = imprint.log.getLogger(__name__)


class HeartbeatThread:
    def __init__(
        self, redis_con, job_id, worker_id, lock_timeout=30, heartbeat_sleep=5
    ):
        self.redis_con = redis_con
        self.job_id = job_id
        self.worker_id = worker_id
        self.lock_timeout = lock_timeout
        self.heartbeat_sleep = heartbeat_sleep
        self.stop = asyncio.Event()
        self.extend_count = 0

    def _refresh(self):
        while not self.stop.is_set():
            self.worker_heartbeat_lock.extend(self.lock_timeout, replace_ttl=True)
            self.extend_count += 1
            logger.debug(
                f"Extended lock {self.lock_name} for {self.lock_timeout} seconds."
            )
            start_sleep = time.time()
            while not self.stop.is_set():
                time.sleep(min(0.1, self.heartbeat_sleep))
                if time.time() - start_sleep > self.heartbeat_sleep:
                    break

    def _cleanup(self):
        success = self.redis_con.srem(f"{self.job_id}:workers", self.worker_id)
        if success:
            logger.debug(f"Removed worker {self.worker_id} from workers set.")
        else:
            logger.warning(
                f"Failed to remove worker {self.worker_id} from workers set."
            )
        if self.worker_heartbeat_lock.owned():
            success = self.worker_heartbeat_lock.release()
            logger.debug(f"Successfully released lock {self.lock_name}.")

    async def __aenter__(self):
        self.lock_name = f"{self.job_id}:heartbeat:{self.worker_id}"
        self.worker_heartbeat_lock = redis.lock.Lock(
            self.redis_con,
            self.lock_name,
            timeout=self.lock_timeout,
            blocking_timeout=1,
            thread_local=False,
        )
        try:
            success = self.worker_heartbeat_lock.acquire()
            if not success:
                raise RuntimeError(
                    f"Heartbeat failed to acquire lock {self.lock_name}."
                )
            logger.debug(f"Acquired lock {self.lock_name}")

            out = self.redis_con.sadd(f"{self.job_id}:workers", self.worker_id)
            if not out:
                raise RuntimeError(f"Worker {self.worker_id} already in workers set.")
            logger.debug(f"Added worker {self.worker_id} to workers set.")

            self.task = asyncio.create_task(asyncio.to_thread(self._refresh))
            return self
        except:  # noqa
            self._cleanup()
            raise

    async def __aexit__(self, *args):
        try:
            self.stop.set()
            await self.task
        finally:
            self._cleanup()
