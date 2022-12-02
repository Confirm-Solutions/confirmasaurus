import os
import time

import awsbatch_tool
import pandas as pd

import confirm


@awsbatch_tool.include_package(confirm)
def f():
    import confirm
    from confirm.imprint import cartesian_grid, ada_tune, hypo

    start = time.time()
    os.system("nvidia-smi")
    print("Loaded confirm in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    g = cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[hypo("x0 < 0")])
    print("Created grid in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    iter, reports, ada = ada_tune(confirm.models.ztest.ZTest1D, g=g, nB=5)
    print("Ran ada in {:.2f} seconds".format(time.time() - start))
    print(pd.DataFrame(reports))


def main():
    awsbatch_tool.local_test(f)


if __name__ == "__main__":
    main()
