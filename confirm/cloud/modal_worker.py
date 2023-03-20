from ..adagrid.backend import LocalBackend
from .modal_backend import process_tiles_config
from .modal_backend import stub


class ModalWorker:
    """
    See ModalBackend for an explanation of why this is in a separate module.
    """

    async def setup(self, worker_args):
        if not hasattr(self, "algo"):
            (model_type, model_args, model_kwargs, algo_type, cfg) = worker_args
            model = model_type(*model_args, **model_kwargs)
            self.algo = algo_type(model, None, None, cfg, None)

    @stub.function(**process_tiles_config)
    async def process_tiles(self, worker_args, tiles_df):
        await self.setup(worker_args)
        lb = LocalBackend()
        async with lb.setup(self.algo):
            return await lb.process_tiles(tiles_df)
