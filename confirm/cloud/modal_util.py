import subprocess
from pathlib import Path

import dotenv
import modal

import imprint as ip

logger = ip.getLogger(__name__)


def get_image(dependency_groups=["cloud"]):
    poetry_dir = Path(__file__).resolve().parent.parent.parent

    context_files = {
        "/.pyproject.toml": str(poetry_dir.joinpath("pyproject.toml")),
        "/.poetry.lock": str(poetry_dir.joinpath("poetry.lock")),
        "/.requirements-jax-cuda.txt": str(
            poetry_dir.joinpath("requirements-jax-cuda.txt")
        ),
        "/.test_secrets.enc.env": str(poetry_dir.joinpath("test_secrets.enc.env")),
    }

    dockerfile_commands = [
        """
        RUN apt-get update && \\
            apt-get install -y golang && \\
            mkdir /go && \\
            export GOPATH=/go && \\
            go install go.mozilla.org/sops/cmd/sops@latest
        """,
        # Modal doesn't support ENV yet so we COPY sops. Might be nice to
        # replace the COPY with this once it is supported.
        # "ENV PATH=/go/bin:$PATH",
        # "COPY /go/bin/sops /usr/bin/sops",
        "COPY /.test_secrets.enc.env /root/test_secrets.enc.env",
        "RUN python -m pip install poetry",
        "COPY /.poetry.lock /tmp/poetry/poetry.lock",
        "COPY /.pyproject.toml /tmp/poetry/pyproject.toml",
        "COPY /.requirements-jax-cuda.txt /tmp/poetry/requirements-jax-cuda.txt",
        f"""
        RUN cd /tmp/poetry && \\
            poetry config virtualenvs.create false && \\
            poetry install --with={','.join(dependency_groups)} --no-root && \\
            pip install --upgrade -r requirements-jax-cuda.txt
        """,
    ]

    return modal.Image.from_dockerhub(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        setup_commands=[
            "apt-get update",
            "apt-get install -y python-is-python3 python3-pip",
        ],
    ).dockerfile_commands(dockerfile_commands, context_files=context_files)


def get_defaults():
    return dict(
        image=get_image(dependency_groups=["test", "cloud"]),
        retries=0,
        mounts=(modal.create_package_mounts(["confirm", "imprint"])),
        secret=modal.Secret.from_name("kms-sops"),
    )


def modalize(stub, **kwargs):
    def decorator(f):
        p = get_defaults()
        p.update(kwargs)
        return stub.function(
            raw_f=f,
            name=f.__qualname__,
            **p,
        )()

    return decorator


def run_on_modal(f, **kwargs):
    stub = modal.Stub("arbitrary_runner")
    wrapper = modalize(stub, **kwargs)(f)
    with stub.run():
        return wrapper.call()


def setup_env(sops_binary="/go/bin/sops"):
    p = subprocess.run(
        [sops_binary, "-d", "--output", "/root/.env", "/root/test_secrets.enc.env"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    logger.debug("Decrypting secrets. stdout from sops:\n %s", p.stdout.decode("utf-8"))
    env_file = "/root/.env"
    env = dotenv.dotenv_values(env_file)
    logger.debug("Environment variables loaded from %s: %s", env_file, list(env.keys()))
    dotenv.load_dotenv(env_file)

    logger.debug("Enabling 64-bit floats in JAX.")
    from jax.config import config

    config.update("jax_enable_x64", True)
