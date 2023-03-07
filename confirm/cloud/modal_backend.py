import asyncio
import contextlib
import logging
import sys

import modal.aio

from . import modal_util
from confirm.adagrid.backend import Backend

logger = logging.getLogger(__name__)
name = "modal_adagrid"
stub = modal.aio.AioStub(name)
process_packet_config = modal_util.get_defaults()
del process_packet_config["retries"]
run_zone_config = modal_util.get_defaults()


class ModalBackend(Backend):
    def __init__(self, n_zones=1, n_workers=None, coordinate_every=5, gpu="any"):
        if n_workers is None:
            n_workers = n_zones
        super().__init__()
        self.lazy_tasks = []
        self.input_cfg = {
            "coordinate_every": coordinate_every,
            "n_zones": n_zones,
            "n_workers": n_workers,
            "gpu": gpu,
        }

    @contextlib.asynccontextmanager
    async def setup(self, algo_type, algo, kwargs):
        # The lazy import below is an insane hack that allows us to set modal
        # configuration at runtime:
        # - Modal needs to use decorators for stub.function.
        # - The inputs to these decorators must be known at import time.
        # - We need to set the GPU configuration at runtime.
        # - Thus, we need to import the worker module at runtime.
        # - In addition, overwriting the stub on each call will multiple
        #   ModalBackends with different parameters inside the same Python
        #   process.
        # - But, to do that, we will need to RE-IMPORT the worker module.
        # - Thus, we need to delete the module from sys.modules at the end of
        #   this
        global stub
        global modal_config
        stub = modal.aio.AioStub(name)
        stub.worker_id_queue = modal.aio.AioQueue()
        process_packet_config["gpu"] = self.input_cfg["gpu"]
        process_packet_config["keep_warm"] = self.input_cfg["n_workers"]
        process_packet_config["concurrency_limit"] = self.input_cfg["n_workers"]
        from .modal_worker import ModalWorker

        self.w = ModalWorker()
        async with stub.run() as app:
            await app.worker_id_queue.put_many(list(range(2, 1000)))

            worker_kwargs = kwargs.copy()
            for k in ["db", "g", "backend"]:
                del worker_kwargs[k]
            self.worker_data = {
                "algo_type": algo_type,
                "job_id": algo.db.job_id,
                "kwargs": worker_kwargs,
            }
            yield
        del sys.modules["confirm.cloud.modal_worker"]

    async def process_initial_incompletes(self, incomplete_packets):
        coros = []
        for i in range(incomplete_packets.shape[0]):
            packet = incomplete_packets[i]
            print("launching", i, packet)
            coros.append(self.w.process_packet.call(self.worker_data, packet, None))
        await asyncio.gather(*coros)

    async def run_zones(self, zone_steps, start_step, end_step):
        return await asyncio.gather(
            *[
                self.w.run_zone.call(self.worker_data, zone_id, start_step, end_step)
                for _, zone_id in enumerate(zone_steps)
            ]
        )
