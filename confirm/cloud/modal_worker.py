from ..adagrid.adagrid import LocalBackend
from .modal_backend import process_tiles_config
from .modal_backend import stub
from .modal_util import setup_env


class ModalWorker:
    """
    See ModalBackend for an explanation of why this is in a separate module.
    """

    async def __aexit__(self, *args):
        if hasattr(self, "algo"):
            await self.algo.db.finalize()

    async def setup(self, worker_args):
        if not hasattr(self, "algo"):
            setup_env()
            (
                algo_type,
                model_type,
                model_args,
                model_kwargs,
                null_hypos,
                cfg,
            ) = worker_args
            import confirm.cloud.clickhouse as ch

            db = ch.ClickhouseTiles.connect(
                job_name=cfg["job_name"], service=cfg["clickhouse_service"]
            )
            model = model_type(*model_args, **model_kwargs)
            self.algo = algo_type(model, null_hypos, db, cfg, None)

    @stub.function(**process_tiles_config)
    async def process_tiles(self, worker_args, tiles_df, refine_deepen):
        await self.setup(worker_args)

        lb = LocalBackend()
        lb.algo = self.algo
        out = await lb.submit_tiles(tiles_df, refine_deepen)
        return out
