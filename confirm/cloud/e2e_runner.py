import sys

import modal
import pytest

import confirm.cloud.modal_util as modal_util

stub = modal.Stub("e2e_runner")

img = modal_util.get_image(dependency_groups=["test", "cloud"])


def run_tests():
    exitcode = pytest.main(["tests/test_duckdb.py"])
    return exitcode.value


@stub.function(
    image=img,
    gpu=modal.gpu.A100(),
    retries=0,
    mounts=(
        modal.create_package_mounts(["confirm", "imprint"])
        + [modal.Mount(local_dir="./tests", remote_dir="/root/tests")]
    ),
    timeout=60 * 60 * 1,
)
def run_cloud_tests():
    return run_tests()


if __name__ == "__main__":
    # run_tests()
    with stub.run():
        exitcode = run_cloud_tests.call()
    sys.exit(exitcode)