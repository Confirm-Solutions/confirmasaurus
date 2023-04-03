import asyncio
import contextlib
import logging
import sys

import modal.aio

from . import modal_util
from confirm.adagrid.adagrid import LocalBackend

logger = logging.getLogger(__name__)
name = "modal_adagrid"
stub = modal.aio.AioStub(name)
process_tiles_config = modal_util.get_defaults()
process_tiles_config["timeout"] = 60 * 60 * 1
del process_tiles_config["retries"]


class ModalBackend(LocalBackend):
    def __init__(self, n_workers=1, gpu="any"):
        super().__init__()
        self.use_clickhouse = True
        self.n_workers = n_workers
        self.gpu = gpu

    def get_cfg(self):
        out = super().get_cfg()
        out.update({"n_workers": self.n_workers, "gpu": self.gpu})
        return out

    @contextlib.asynccontextmanager
    async def setup(self, algo):
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
        process_tiles_config["gpu"] = self.gpu
        process_tiles_config["keep_warm"] = self.n_workers
        process_tiles_config["concurrency_limit"] = self.n_workers
        from .modal_worker import ModalWorker

        self.w = ModalWorker()
        async with stub.run() as app:
            await app.worker_id_queue.put_many(list(range(2, self.n_workers * 5)))
            filtered_cfg = {
                k: v for k, v in algo.cfg.items() if k in self.algo_cfg_entries
            }
            self.worker_args = (
                type(algo),
                algo.model_type,
                algo.null_hypos,
                filtered_cfg,
            )
            yield
        del sys.modules["confirm.cloud.modal_worker"]

    async def submit_tiles(self, tiles_df, refine_deepen, report):
        return asyncio.create_task(
            self.w.process_tiles.call(self.worker_args, tiles_df, refine_deepen, report)
        )

    async def wait_for_results(self, awaitable):
        return await awaitable
