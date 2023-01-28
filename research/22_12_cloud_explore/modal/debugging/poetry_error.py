import os
import subprocess

import modal

stub = modal.Stub("jaxtest")

if not stub.is_inside():
    pyproject_toml = """
    [tool.poetry]
    name = "confirm"
    version = "0.1.0"
    description = ""
    authors = ["Confirm Solutions <research@confirmsol.org>"]

    [tool.poetry.dependencies]
    python = "~3.10"
    jax = "^0.3.25"
    jaxlib = "^0.3.25"

    [[tool.poetry.source]]
    name = "jax"
    url = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    default = false
    secondary = false

    [build-system]
    requires = ["poetry-core>=1.0.0"]
    build-backend = "poetry.core.masonry.api"
    """
    should_write = True
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "r") as f:
            should_write = f.read() != pyproject_toml
    if should_write:
        with open("pyproject.toml", "w") as f:
            f.write(pyproject_toml)
        subprocess.run(["poetry", "lock"])

img = modal.Image.from_dockerhub(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    setup_commands=[
        "apt-get update",
        "apt-get install -y python-is-python3 python3-pip",
    ],
).poetry_install_from_file("pyproject.toml")


@stub.function(image=img, gpu=True)
def f(n):
    import jax

    seed = jax.random.PRNGKey(0)
    arr = jax.random.choice(seed, n, shape=(n, n))
    return float(arr.sum())


if __name__ == "__main__":
    with stub.run():
        # outputs 427
        print(f(10))
