import asyncio
import logging

import imprint as ip
from . import modal_util
from .modal_backend import modal_config
from .modal_backend import stub
from confirm.adagrid.db import DatabaseLogging
from confirm.adagrid.init import init
from confirm.adagrid.step import new_step
from confirm.adagrid.step import process_packet

logger = logging.getLogger(__name__)


class ModalWorker:
    def __init__(self):
        self.initialized = False

    async def setup(self, *, algo_type, job_id, kwargs, worker_id_queue):
        from . import clickhouse as ch

        if not self.initialized:
            modal_util.setup_env()
            ip.configure_logging()
            db = ch.Clickhouse.connect(job_id, no_create=True)
            worker_kwargs = kwargs.copy()
            worker_kwargs["db"] = db
            with DatabaseLogging(db=db):
                worker_id = await worker_id_queue.get(block=False)
                if worker_id is None:
                    raise RuntimeError("No worker ID available")
                self.algo, _, _ = await init(
                    algo_type, False, worker_id, None, worker_kwargs
                )
                self.initialized = True

    @stub.function(**modal_config)
    async def process_packet(self, worker_cfg, packet):
        await self.setup(**worker_cfg)
        with DatabaseLogging(db=self.algo.db):
            insert_task, report_task = await process_packet(self.algo, *packet)
            await insert_task
            await report_task

    @stub.function(**modal_config)
    async def run_zone(self, worker_cfg, zone_id, start_step, end_step):
        await self.setup(**worker_cfg)
        with DatabaseLogging(db=self.algo.db):
            if start_step >= end_step:
                return
            logger.debug(
                f"Zone {zone_id} running from step {start_step} "
                f"through step {end_step - 1}."
            )
            for step_id in range(start_step, end_step):
                status, n_packets, lazy_tasks = await new_step(
                    self.algo, zone_id, step_id
                )
                lazy_tasks.extend(lazy_tasks)
                coros = []
                for i in range(n_packets):
                    packet = (zone_id, step_id, i)
                    coros.append(self.process_packet.call(worker_cfg, packet))
                await asyncio.gather(*coros)
                if status.done():
                    logger.debug(f"Zone {zone_id} finished with status {status}.")
                    break
            await asyncio.gather(*lazy_tasks)
            return status
