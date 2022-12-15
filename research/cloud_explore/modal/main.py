import time

import modal

stub = modal.Stub("confirm")

context_files = {"/.pyproject.toml": "pyproject.toml", "/.poetry.lock": "poetry.lock"}

dockerfile_commands = [
    "RUN python -m pip install poetry",
    "COPY /.poetry.lock /tmp/poetry/poetry.lock",
    "COPY /.pyproject.toml /tmp/poetry/pyproject.toml",
    "RUN cd /tmp/poetry && \\ ",
    "  poetry config virtualenvs.create false && \\ ",
    "  poetry install --with=cloud --no-root",
]

img = modal.Image.from_dockerhub(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    setup_commands=[
        "apt-get update",
        "apt-get install -y python-is-python3 python3-pip",
    ],
).dockerfile_commands(dockerfile_commands, context_files=context_files)
# .poetry_install_from_file("pyproject.toml", "poetry.lock")


@stub.function(
    image=img,
    # gpu=True,
    gpu=modal.gpu.A100(),
    retries=0,
    mounts=modal.create_package_mounts(["confirm"]),
)
def f(n):
    start = time.time()
    import confirm.imprint as ip
    from confirm.models.ztest import ZTest1D
    import pandas as pd
    import os

    os.system("nvidia-smi")
    print("Loaded confirm in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    print("Created grid in {:.2f} seconds".format(time.time() - start))
    start = time.time()
    iter, reports, ada = ip.ada_calibrate(ZTest1D, g=g, nB=5)
    print("Ran ada in {:.2f} seconds".format(time.time() - start))
    print(pd.DataFrame(reports))


if __name__ == "__main__":
    with stub.run():
        # outputs 427
        f(10)
