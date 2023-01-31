"""
This file is used to run tests in the cloud using Modal.
"""
import sys

import dotenv
import modal
import pytest

import confirm.cloud.modal_util as modal_util

# Load environment variables from .env file before using modal

dotenv.load_dotenv()

stub = modal.Stub("test_runner")

img = modal_util.get_image(dependency_groups=["test", "cloud"])


def run_tests(argv=None):
    if argv is None:
        argv = []
    print(argv)

    # This line allows us to launch a second Modal app from within a Modal app.
    modal.app._is_container_app = False

    exitcode = pytest.main(argv)
    print(exitcode)
    return exitcode.value


@stub.function(
    image=img,
    gpu=modal.gpu.A100(),
    retries=0,
    mounts=(
        modal.create_package_mounts(["confirm", "imprint"])
        + [
            modal.Mount(local_dir="./tests", remote_dir="/root/tests"),
            modal.Mount(local_dir="./imprint/tests", remote_dir="/root/imprint/tests"),
            modal.Mount(
                local_dir="./imprint/tutorials", remote_dir="/root/imprint/tutorials"
            ),
            modal.Mount(local_dir="./", remote_dir="/root", recursive=False),
        ]
    ),
    timeout=60 * 60 * 1,
    secrets=[modal.Secret.from_name("kms-sops")],
)
def run_cloud_tests(argv=None):
    modal_util.decrypt_secrets()
    return run_tests(argv=argv)


if __name__ == "__main__":
    argv = None if len(sys.argv) == 1 else sys.argv[1:]
    with stub.run():
        exitcode = run_cloud_tests.call(argv)
    sys.exit(exitcode)
