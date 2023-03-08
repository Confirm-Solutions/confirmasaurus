import asyncio
import logging
from contextlib import ExitStack

import modal

import imprint as ip
from . import modal_util
from .modal_backend import process_packet_config
from .modal_backend import run_zone_config
from .modal_backend import stub
from confirm.adagrid.db import DatabaseLogging
from confirm.adagrid.init import init
from confirm.adagrid.step import new_step
from confirm.adagrid.step import process_packet

logger = logging.getLogger(__name__)


class ModalWorker:
    """
    See ModalBackend for an explanation of why this is in a separate module.

    This class has the various functions that can run in the cloud by Modal.
    """

    def __init__(self):
        self.initialized = False

    def __enter__(self):
        self.worker_id = modal.container_app.worker_id_queue.get(block=False)
        if self.worker_id is None:
            raise RuntimeError("No worker ID available")
        ip.log.worker_id.set(self.worker_id)
        modal_util.setup_env()

    def __exit__(self, *args):
        if hasattr(self, "_stack"):
            self._stack.__exit__(*args)

    async def setup(self, *, algo_type, job_id, kwargs):
        if not self.initialized:
            from . import clickhouse as ch

            db = ch.Clickhouse.connect(job_id, no_create=True)
            worker_kwargs = kwargs.copy()
            worker_kwargs["db"] = db
            self.db_logging = DatabaseLogging(db=db)
            with ExitStack() as stack:
                stack.enter_context(self.db_logging)
                self.algo, _, _ = await init(
                    algo_type, False, self.worker_id, None, worker_kwargs
                )
                self.initialized = True
                self._stack = stack.pop_all()

    @stub.function(**process_packet_config)
    async def process_packet(self, worker_cfg, packet, packet_df):
        await self.setup(**worker_cfg)
        if packet is None:
            zone_id = packet_df.iloc[0]["zone_id"]
            step_id = packet_df.iloc[0]["step_id"]
            packet_id = packet_df.iloc[0]["packet_id"]
        else:
            zone_id, step_id, packet_id = packet
        insert_task, report_task = await process_packet(
            self.algo, zone_id, step_id, packet_id, packet_df=packet_df
        )
        await insert_task
        # TODO: can we make this lazy
        await report_task

    @stub.function(**run_zone_config)
    async def run_zone(self, worker_cfg, zone_id, start_step, end_step):
        await self.setup(**worker_cfg)
        if start_step >= end_step:
            return
        logger.debug(
            f"Zone {zone_id} running from step {start_step} "
            f"through step {end_step - 1}."
        )
        for step_id in range(start_step, end_step):
            status, tiles_df, before_next_step_tasks, lazy_tasks = await new_step(
                self.algo, zone_id, step_id
            )
            lazy_tasks.extend(lazy_tasks)
            if tiles_df is not None and tiles_df.shape[0] > 0:
                coros = []
                for _, packet_df in tiles_df.groupby("packet_id"):
                    logger.debug(
                        f"Zone {zone_id} launching the processing of packet "
                        f"{packet_df.iloc[0]['packet_id']}."
                    )
                    coros.append(self.process_packet.call(worker_cfg, None, packet_df))
                await asyncio.gather(*coros)
            await asyncio.gather(*before_next_step_tasks)
            if status.done():
                logger.debug(f"Zone {zone_id} finished with status {status}.")
                break
        await asyncio.gather(*lazy_tasks)
        return status
