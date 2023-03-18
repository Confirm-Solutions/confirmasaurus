import confirm.adagrid as ada
import confirm.cloud.coiled_backend as coiled_backend
import confirm.models.wd41 as wd41
import imprint as ip


def benchmark():
    import os
    import time

    model = wd41.WD41(0, 1, ignore_intersection=True)
    grid = ip.cartesian_grid(
        [-2.5, -2.5, -2.5, -2.5],
        [1.0, 1.0, 1.0, 1.0],
        n=[10, 10, 10, 10],
        null_hypos=model.null_hypos,
    )
    total_sims = grid.df.shape[0] * 2**14
    print(total_sims)
    os.system("nvidia-smi")
    # T4: ~24ns per sim
    # V100: 20ns per sim
    # A10G: 17.5ns per sim
    # A100: 16ns per sim
    for i in range(3):
        start = time.time()
        cal_df = ip.calibrate(wd41.WD41, g=grid, alpha=0.025, K=2**14)
        print(cal_df["lams"].min())
        R = time.time() - start
        n_per_sec = total_sims / R
        ns_per_sim = 1 / n_per_sec * 1e9
        print(R, ns_per_sim)


# if __name__ == "__main__":
#     from confirm.cloud.coiled_backend import setup_cluster
#     cluster = setup_cluster()
#     client = cluster.get_client()
#     client.submit(benchmark).result()


def main():
    ip.package_settings()
    model = wd41.WD41(0, 1, ignore_intersection=True)
    grid = ip.cartesian_grid(
        [-2.5, -2.5, -2.5, -2.5],
        [1.0, 1.0, 1.0, 1.0],
        n=[10, 10, 10, 10],
        null_hypos=model.null_hypos,
    )
    db = ada.ada_calibrate(  # noqa
        wd41.WD41,
        g=grid,
        alpha=0.025,
        bias_target=0.001,
        grid_target=0.001,
        std_target=0.002,
        n_K_double=6,
        calibration_min_idx=80,
        step_size=2**18,
        packet_size=2**14,
        model_kwargs={"ignore_intersection": True},
        n_zones=4,
        backend=coiled_backend.CoiledBackend(n_workers=16),
        job_name="wd41_4d_v3",
    )


if __name__ == "__main__":
    main()
