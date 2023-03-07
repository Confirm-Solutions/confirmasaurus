import dotenv
import modal

import confirm.cloud.modal_util as modal_util

dotenv.load_dotenv()

# def benchmark():
# import os
# import time
# model = wd41.WD41(0, 1, ignore_intersection=True)
# grid = ip.cartesian_grid(
#     [-2.5, -2.5, -2.5, -2.5],
#     [1.0, 1.0, 1.0, 1.0],
#     n=[10, 10, 10, 10],
#     null_hypos=model.null_hypos,
# )
#     total_sims = grid.df.shape[0] * 2**14
#     print(total_sims)
#     os.system('nvidia-smi')
#     # T4: ~24ns per sim
#     # V100: 20ns per sim
#     # A10G: 17.5ns per sim
#     # A100: 16ns per sim
#     for i in range(3):
#         start = time.time()
#         cal_df = ip.calibrate(wd41.WD41, g=grid, alpha=0.025, K=2**14)
#         print(cal_df["lams"].min())
#         R = time.time() - start
#         n_per_sec = total_sims / R
#         ns_per_sim = 1 / n_per_sec * 1e9
#         print(R, ns_per_sim)

# if __name__ == "__main__":
#     import confirm.cloud.modal_util as modal_util
#     modal_util.run_on_modal(benchmark, gpu='A10G')


stub = modal.Stub("WD41")


# running in detached mode:
# modal run --detach research/closedtesting/wd41_4d.py::main
@stub.function(**modal_util.get_defaults())
def main():
    import imprint as ip
    import confirm.cloud.clickhouse as ch
    import confirm.models.wd41 as wd41

    from confirm.adagrid import ada_calibrate
    from confirm.cloud.modal_backend import ModalBackend

    modal_util.setup_env()

    # This line allows us to launch a second Modal app from within a Modal app.
    modal.app._is_container_app = False

    model = wd41.WD41(0, 1, ignore_intersection=True)
    grid = ip.cartesian_grid(
        [-2.5, -2.5, -2.5, -2.5],
        [1.0, 1.0, 1.0, 1.0],
        n=[10, 10, 10, 10],
        null_hypos=model.null_hypos,
    )
    db = ch.Clickhouse.connect()
    ada_calibrate(
        wd41.WD41,
        g=grid,
        db=db,
        alpha=0.025,
        bias_target=0.005,
        grid_target=0.005,
        std_target=0.01,
        n_K_double=6,
        calibration_min_idx=80,
        step_size=2**16,
        packet_size=2**13,
        model_kwargs={"ignore_intersection": True},
        n_steps=3,
        backend=ModalBackend(n_zones=4, n_workers=10, coordinate_every=1, gpu="T4"),
    )
    print(db.job_id)


if __name__ == "__main__":
    with stub.run():
        main.call()
