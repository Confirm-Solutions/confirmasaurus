import time

import confirm.models.wd41 as wd41
import imprint as ip

model = wd41.WD41(0, 1, ignore_intersection=True)
grid = ip.cartesian_grid(
    [-2.5, -2.5, -2.5, -2.5],
    [1.0, 1.0, 1.0, 1.0],
    n=[10, 10, 10, 10],
    null_hypos=model.null_hypos,
)

total_sims = grid.df.shape[0] * 2**14
print(total_sims)

for i in range(3):
    start = time.time()
    cal_df = ip.calibrate(wd41.WD41, g=grid, alpha=0.025, K=2**14)
    print(cal_df["lams"].min())
    R = time.time() - start
    n_per_sec = total_sims / R
    ns_per_sim = 1 / n_per_sec * 1e9
    print(R, ns_per_sim)

# from confirm.adagrid import ada_calibrate
# db = ada_calibrate(
#     wd41.WD41,
#     g=grid,
#     alpha=0.025,
#     bias_target=0.005,
#     grid_target=0.005,
#     std_target=0.01,
#     n_K_double=6,
#     calibration_min_idx=80,
#     step_size=2**16,
#     packet_size=2**13,
#     model_kwargs={"ignore_intersection": True},
#     n_steps=2,
# )
