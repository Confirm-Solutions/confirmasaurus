from ..adagrid.adagrid import LocalBackend
from ..adagrid.adagrid import setup_db
from .modal_backend import process_tiles_config
from .modal_backend import stub
from .modal_util import setup_env


class ModalWorker:
    """
    See ModalBackend for an explanation of why this is in a separate module.
    """

    async def __aexit__(self, *args):
        if hasattr(self, "algo"):
            await self.algo.db.ch_wait()

    async def setup(self, worker_args):
        if not hasattr(self, "algo"):
            setup_env()
            (model_type, model_args, model_kwargs, algo_type, cfg) = worker_args
            db = setup_db(cfg)
            model = model_type(*model_args, **model_kwargs)
            self.algo = algo_type(model, None, db, cfg, None)

    @stub.function(**process_tiles_config)
    async def process_tiles(self, worker_args, tiles_df, refine_deepen):
        await self.setup(worker_args)

        lb = LocalBackend()
        lb.algo = self.algo
        return lb.submit_tiles(tiles_df, refine_deepen)
