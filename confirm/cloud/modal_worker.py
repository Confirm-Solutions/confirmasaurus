import jax

from .modal_backend import process_tiles_config
from .modal_backend import stub


class ModalWorker:
    """
    See ModalBackend for an explanation of why this is in a separate module.

    This class has the various functions that can run in the cloud by Modal.
    """

    def __init__(self):
        self.initialized = False

    def __exit__(self, *args):
        if hasattr(self, "_stack"):
            self._stack.__exit__(*args)

    async def setup(self, worker_args):
        if not self.initialized:
            (model_type, model_args, model_kwargs, algo_type, cfg) = worker_args
            model = model_type(*model_args, **model_kwargs)
            self.algo = algo_type(model, None, None, cfg, None)

    @stub.function(**process_tiles_config)
    async def process_tiles(self, worker_args, tiles_df):
        await self.setup(worker_args)
        tbs = self.algo.cfg["tile_batch_size"]
        if tbs is None:
            tbs = dict(gpu=64, cpu=4)[jax.lib.xla_bridge.get_backend().platform]
        return await self.algo.process_tiles(tiles_df=tiles_df, tile_batch_size=tbs)
