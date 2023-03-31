import modal

import confirm.cloud.modal_util as modal_util

stub = modal.Stub("wd41_leader")
modal_cfg = modal_util.get_defaults()

# 8 cores and 100 GB of memory
modal_cfg["cpu"] = 8
modal_cfg["memory"] = 1024 * 64
modal_cfg["timeout"] = 60 * 60 * 24


def explore_cpu():
    def toytoy(cmd):
        import os

        try:
            for i in range(6):
                print(cmd)
            os.system(cmd)
        except:  # noqa
            pass

    toytoy("cat /proc/cpuinfo")
    toytoy("lscpu")


@stub.function(**modal_cfg)
def main():
    import imprint as ip

    import os

    os.system("pip install line_profiler")
    import line_profiler
    import builtins

    prof = line_profiler.LineProfiler()
    builtins.__dict__["profile"] = prof

    try:
        import dotenv

        if stub.is_inside():
            modal_util.setup_env()
            modal_util.coiled_login()
            # import os
            # import dask
            # dask.config.set({"coiled.token": os.environ["COILED_TOKEN"]})
        else:
            ip.package_settings()
            dotenv.load_dotenv()

        import confirm.adagrid as ada

        import confirm.cloud.coiled_backend as coiled_backend
        import confirm.models.wd41 as wd41

        model = wd41.WD41(0, 1, ignore_intersection=True)
        grid = ip.cartesian_grid(
            [-2.0, -2.0, -2.0, -2.0],
            [1.0, 1.0, 1.0, 1.0],
            n=[10, 10, 10, 10],
            null_hypos=model.null_hypos,
        )

        db = ada.ada_calibrate(  # noqa
            wd41.WD41,
            job_name="wd41_4d_v92",
            clickhouse_service="PROD",
            g=grid,
            alpha=0.025,
            bias_target=0.001,
            grid_target=0.001,
            std_target=0.002,
            n_K_double=6,
            calibration_min_idx=70,
            step_size=2**16,
            packet_size=2**13,
            n_parallel_steps=2,
            model_kwargs={"ignore_intersection": True},
            backend=coiled_backend.CoiledBackend(
                update_software_env=False, restart_workers=True, n_workers=16
            ),
            n_steps=8,
        )
        print("Done")
    finally:
        prof.print_stats(output_unit=1e-3)


if __name__ == "__main__":
    main()
