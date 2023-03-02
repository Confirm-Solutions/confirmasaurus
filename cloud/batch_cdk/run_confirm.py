import os
import time

import awsbatch_tool
import pandas as pd

import confirm


@awsbatch_tool.include_package(confirm)
def f():
    import confirm.adagrid as ip
    from imprint.models.ztest import ZTest1D

    os.system("nvidia-smi")
    start = time.time()
    print("Loaded confirm in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    print("Created grid in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    db = ip.ada_calibrate(ZTest1D, g=g, nB=5)
    print("Ran ada in {:.2f} seconds".format(time.time() - start))
    print(pd.DataFrame(db.get_reports()))


def main():
    # awsbatch_tool.local_test(f)
    resp, bucket, filename = awsbatch_tool.remote_run(
        f, cpus=2, memory=2**12, gpu=True
    )
    print(resp)
    print(bucket)
    print(filename)


if __name__ == "__main__":
    main()
