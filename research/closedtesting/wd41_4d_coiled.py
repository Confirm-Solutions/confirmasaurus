import modal

import confirm.cloud.modal_util as modal_util

stub = modal.Stub("wd41_leader")
modal_cfg = modal_util.get_defaults()

# 4 cores and 64 GB of memory
modal_cfg["cpu"] = 4
modal_cfg["memory"] = 1024 * 64
modal_cfg["timeout"] = 60 * 60 * 24


@stub.function(**modal_cfg)
def main():
    import coiled
    import confirm.adagrid as ada
    import confirm.cloud.coiled_backend as coiled_backend
    import confirm.models.wd41 as wd41

    import imprint as ip
    import dotenv

    if stub.is_inside():
        modal_util.setup_env()
        # modal_util.coiled_login()
        import os
        import dask

        dask.config.set({"coiled.token": os.environ["COILED_TOKEN"]})
    else:
        ip.package_settings()
        dotenv.load_dotenv()

    model = wd41.WD41(0, 1, ignore_intersection=True)
    grid = ip.cartesian_grid(
        [-2.0, -2.0, -2.0, -2.0],
        [1.0, 1.0, 1.0, 1.0],
        n=[10, 10, 10, 10],
        null_hypos=model.null_hypos,
    )
    db = ada.ada_calibrate(  # noqa
        wd41.WD41,
        job_name="wd41_4d_v7",
        g=grid,
        alpha=0.025,
        bias_target=0.001,
        grid_target=0.001,
        std_target=0.002,
        n_K_double=6,
        calibration_min_idx=70,
        step_size=2**18,
        packet_size=2**14,
        model_kwargs={"ignore_intersection": True},
        n_zones=4,
        backend=coiled_backend.CoiledBackend(
            restart_workers=True,
            n_workers=16,
            cluster=coiled.Cluster(name="confirm-coiled"),
        ),
        n_steps=3,
    )


if __name__ == "__main__":
    main()
