import os

import dotenv

dotenv.load_dotenv()

env = {k: v for k, v in os.environ.items() if k.startswith("CLICKHOUSE")}


def f():
    import imprint as ip
    import confirm.adagrid as ada
    import confirm.models.wd41 as wd41

    # from dask.distributed import get_client

    os.environ.update(env)
    ip.package_settings()

    model = wd41.WD41(0, 1, ignore_intersection=True)
    grid = ip.cartesian_grid(
        [-2.0, -2.0, -2.0, -2.0],
        [1.0, 1.0, 1.0, 1.0],
        n=[10, 10, 10, 10],
        null_hypos=model.null_hypos,
    )

    import confirm.cloud.coiled_backend as coiled_backend

    db = ada.ada_calibrate(  # noqa
        wd41.WD41,
        job_name="wd41_4d",
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
        # backend=coiled_backend.CoiledBackend(client=get_client()),
        backend=coiled_backend.CoiledBackend(n_workers=16),
        # TODO:
        # TODO:
        # TODO:
        n_steps=8,
        nB=1,
    )
    print("Done inner")


def main():
    import confirm.cloud.coiled_backend as coiled_backend

    cluster, client = coiled_backend.setup_cluster(n_workers=16)
    client.submit(f).result()
    print("Done outer")


if __name__ == "__main__":
    # main()
    f()
