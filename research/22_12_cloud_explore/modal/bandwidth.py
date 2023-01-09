import time

import modal
import numpy as np

stub = modal.Stub("bandwidthtest")

volume = modal.SharedVolume().persist("bandwidthtest_data")


@stub.function(shared_volumes={"/data": volume})
def save():
    start = time.time()
    data = np.random.uniform(size=(256000, 350 * 4))
    print("generate", time.time() - start)
    start = time.time()
    np.save("/data/data.npy", data)
    print("save", time.time() - start)


@stub.function(shared_volumes={"/data": volume})
def load(i):
    start = time.time()
    np.load("/data/data.npy")
    print("load", time.time() - start)


with stub.run():
    # save.call()
    list(load.map(range(3)))
