import asyncio
import logging

from confirm.adagrid.convergence import WorkerStatus
from confirm.adagrid.init import init
from confirm.adagrid.step import new_step
from confirm.adagrid.step import process_zone

logger = logging.getLogger(__name__)


class LocalBackend:
    def run(self, algo_type, kwargs):
        return maybe_start_event_loop(self._run_async(algo_type, kwargs))

    async def _run_async(self, algo_type, kwargs):
        algo, zone_info = await init(algo_type, 1, kwargs)
        await asyncio.gather(
            *[
                self._run_zone(algo, zone_id, zone_info[zone_id])
                for zone_id in zone_info
            ]
        )
        return algo.db

    async def _run_zone(self, algo, zone_id, n_packets):
        # TODO: get initial step id and zone info
        insert_reports = []
        for step_id in range(algo.cfg["n_steps"]):
            insert_reports.extend(await process_zone(algo, zone_id, step_id, n_packets))

            status, n_packets, report_task = await new_step(algo, zone_id, step_id + 1)
            insert_reports.append(report_task)
            if (
                status == WorkerStatus.CONVERGED
                or status == WorkerStatus.EMPTY_STEP
                or status == WorkerStatus.REACHED_N_STEPS
            ):
                logger.debug(f"Zone {zone_id} finished with status {status}.")
                break
        await asyncio.gather(*insert_reports)


def maybe_start_event_loop(coro):
    # Check if an event loop is already running. If we're running through
    # Jupyter, then an event loop will already be running and we're not allowed
    # to start a new event loop inside of the existing one.
    try:
        loop = asyncio.get_running_loop()
        existing_loop = True
    except RuntimeError:
        existing_loop = False

    # We run here instead of under the except statement so that any
    # exceptions are not prefixed with "During handling of the above
    # exception..." messages.
    if existing_loop:
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)


class MultiprocessingBackend:
    def __init__(self, n_workers=1):
        self.n_workers = n_workers

    #     async def run(self, ada):
    #         import confirm.cloud.clickhouse as ch

    #         model_type = ada.model_type
    #         algo_type = ada.algo_type
    #         callback = ada.callback
    #         job_id = ada.db.job_id

    #         def _worker():
    #             db = ch.Clickhouse.connect(job_id=job_id)
    #             runner = Adagrid(model_type, None, db, algo_type, callback,
    #                dict(), dict())
    #             runner._run_local()

    async def run(self, ada, initial_worker_ids):
        import modal
        import confirm.cloud.clickhouse as ch
        import confirm.cloud.modal_util as modal_util

        assert len(initial_worker_ids) == self.n_workers

        job_id = ada.db.job_id
        algo_type = ada.algo_type
        pass_params = dict(
            callback=ada.callback,
            model_type=ada.model_type,
            overrides=dict(),
            backend=LocalBackend(),
            g=None,
        )

        stub = modal.Stub(f"{self.job_name_prefix}_{job_id}")

        p = modal_util.get_defaults()
        p.update(dict(gpu=self.gpu, serialized=True))

        @stub.function(**p)
        def _modal_adagrid_worker(i):
            modal_util.setup_env()
            kwargs = copy.deepcopy(pass_params)
            kwargs["db"] = ch.Clickhouse.connect(job_id=job_id)
            kwargs["worker_id"] = initial_worker_ids[i]
            asyncio.run(init_and_run(algo_type, **kwargs))

        logger.info(f"Launching Modal job with {self.n_workers} workers")
        with stub.run(show_progress=False):
            _ = list(_modal_adagrid_worker.map(range(self.n_workers)))
        return ada.db
