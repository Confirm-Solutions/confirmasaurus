import asyncio
import logging

import modal.aio
import numpy as np

import confirm.cloud.modal_util as modal_util
from .backend import get_next_coord
from .backend import maybe_start_event_loop
from confirm.adagrid.coordinate import coordinate
from confirm.adagrid.init import init
from confirm.adagrid.step import new_step
from confirm.adagrid.step import process_packet

logger = logging.getLogger(__name__)

stub = modal.aio.AioStub()
stub.worker_id_queue = modal.aio.AioQueue()
modal_config = modal_util.get_defaults()


class ModalWorker:
    def __init__(self):
        self.initialized = False

    async def setup(self, *, algo_type, job_id, kwargs, worker_id_queue):
        import confirm.cloud.clickhouse as ch

        if not self.initialized:
            modal_util.setup_env()
            kwargs["db"] = ch.Clickhouse.connect(job_id, no_create=True)
            worker_id = await worker_id_queue.get(block=False)
            if worker_id is None:
                raise RuntimeError("No worker ID available")
            self.algo, _, _ = await init(algo_type, False, worker_id, None, kwargs)
            self.initialized = True

    @stub.function(**modal_config)
    async def process_packet(self, worker_cfg, packet):
        await self.setup(**worker_cfg)
        insert_task, report_task = await process_packet(self.algo, *packet)
        await insert_task
        await report_task

    @stub.function(**modal_config)
    async def run_zone(self, worker_cfg, zone_id, start_step_id, end_step_id):
        await self.setup(**worker_cfg)
        if start_step_id >= end_step_id:
            return
        logger.debug(
            f"Zone {zone_id} running from step {start_step_id} "
            f"through step {end_step_id - 1}."
        )
        for step_id in range(start_step_id, end_step_id):
            status, n_packets, lazy_tasks = await new_step(self.algo, zone_id, step_id)
            lazy_tasks.extend(lazy_tasks)
            coros = []
            for i in range(n_packets):
                packet = (zone_id, step_id, i)
                coros.append(self.process_packet(worker_cfg, packet))
            await asyncio.gather(*coros)
            if status.done():
                logger.debug(f"Zone {zone_id} finished with status {status}.")
                break
        await asyncio.gather(*lazy_tasks)
        return status


class ModalBackend:
    def __init__(self, n_zones=1, coordinate_every=5, gpu="any"):
        self.lazy_tasks = []
        self.input_cfg = {
            "coordinate_every": coordinate_every,
            "n_zones": n_zones,
            "gpu": gpu,
        }
        self.already_run = False

    def run(self, algo_type, kwargs):
        if self.already_run:
            raise RuntimeError("ModalBackend can only be used once")
        self.already_run = True
        kwargs.update(self.input_cfg)
        return maybe_start_event_loop(self._run_async(algo_type, kwargs))

    async def _run_async(self, algo_type, kwargs):
        # it's okay to use input_cfg['n_zones'] here because it's only used at
        # the start of a job.
        algo, incomplete_packets, zone_steps = await init(
            algo_type, True, 1, self.input_cfg["n_zones"], kwargs
        )

        w = ModalWorker()

        incomplete_packets = np.array(incomplete_packets)
        min_step_completed = min(zone_steps.values())
        max_step_completed = max(zone_steps.values())
        every = algo.cfg["coordinate_every"]
        next_coord = get_next_coord(min_step_completed, every)
        assert next_coord == get_next_coord(max_step_completed, every)
        first_step = min_step_completed + 1
        n_zones = algo.cfg["n_zones"]

        async with stub.run() as app:
            await app.worker_id_queue.put_many(list(range(1000)))

            worker_kwargs = kwargs.copy()
            for k in ["db", "g", "backend"]:
                del worker_kwargs[k]
            worker_data = {
                "algo_type": algo_type,
                "job_id": algo.db.job_id,
                "kwargs": worker_kwargs,
                "worker_id_queue": app.worker_id_queue,
            }

            coros = []
            for i in range(incomplete_packets.shape[0]):
                packet = incomplete_packets[i]
                # w = workers[i % len(workers)]
                # coros.append(await w.process_packet.call(packet))
                print("launching", i, packet)
                coros.append(w.process_packet.call(worker_data, packet))
            await asyncio.gather(*coros)

            while first_step < algo.cfg["n_steps"]:
                last_step = min(next_coord, algo.cfg["n_steps"])
                statuses = await asyncio.gather(
                    *[
                        w.run_zone.call(worker_data, zone_id, first_step, last_step + 1)
                        for _, zone_id in enumerate(zone_steps)
                    ]
                )

                # If there's only one zone and that zone is done, then we're
                # totally done.
                if len(statuses) == 1:
                    if statuses[0].done():
                        break

                if n_zones > 1:
                    # If there's more than one zone, then we need to coordinate.
                    # NOTE: coordinations happen *before* the same-named step.
                    # e.g. a coordination at step 5 happens before new_step and
                    # process_packets for 5.
                    coord_status, lazy_tasks, zone_steps = await coordinate(
                        algo, last_step + 1, n_zones
                    )
                    self.lazy_tasks.extend(lazy_tasks)
                    if coord_status.done():
                        break

                first_step = last_step + 1
                next_coord += every

            # TODO: currently we only verify at the end, should we do it more often?
            verify_task = asyncio.create_task(algo.db.verify())

            await asyncio.gather(*self.lazy_tasks)
            self.lazy_tasks = []

            await verify_task
            return algo.db
