import os
import time

import awsbatch_tool
import pandas as pd

import confirm


@awsbatch_tool.include_package(confirm)
def f():
    import confirm.imprint as ip
    from confirm.models.ztest import ZTest1D

    print(os.environ["LD_LIBRARY_PATH"])
    print(os.environ["LD_LIBRARY_PATH"])
    print(os.environ["LD_LIBRARY_PATH"])
    print(os.listdir("/usr/local/cuda/compat"))
    print(os.listdir("/usr/local/cuda/compat"))
    print(os.listdir("/usr/local/cuda/compat"))
    os.system("nvidia-smi")
    os.system("nvidia-smi")
    os.system("nvidia-smi")

    start = time.time()
    print("Loaded confirm in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    print("Created grid in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    iter, reports, ada = ip.ada_tune(ZTest1D, g=g, nB=5)
    print("Ran ada in {:.2f} seconds".format(time.time() - start))
    print(pd.DataFrame(reports))


def main():
    awsbatch_tool.local_test(f)
    # resp, bucket, filename = awsbatch_tool.remote_run(
    #   f2, cpus=2, memory=2**12, gpu=True
    # )
    # print(resp)
    # print(bucket)
    # print(filename)


if __name__ == "__main__":
    main()
