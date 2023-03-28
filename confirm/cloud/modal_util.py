import logging
import os
import subprocess
from pathlib import Path

import dotenv
import modal

logger = logging.getLogger(__name__)


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
        setup_dockerfile_commands=[
            "RUN apt-get update && apt-get install -y python3-pip python-is-python3"
        ],
    ).dockerfile_commands(dockerfile_commands, context_files=context_files)


def include_file(fn):
    return "test_secrets.enc.env" in fn
    # exclude unencrypted secrets
    # return (fn != ".env") and (os.path.getsize(fn) < 1e6)


def get_defaults():
    return dict(
        image=get_image(dependency_groups=["test", "cloud"]),
        retries=0,
        mounts=[
            *modal.create_package_mounts(["confirm", "imprint"]),
            modal.Mount.from_local_dir(
                "./",
                remote_path="/root",
                condition=include_file,
                recursive=False,
            ),
        ],
        secrets=[modal.Secret.from_name("kms-sops")],
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
    if "MODAL_TOKEN_ID" not in os.environ:
        decrypt_secrets()
        dotenv.load_dotenv()
    stub = modal.Stub("arbitrary_runner")
    wrapper = modalize(stub, **kwargs)(f)
    with stub.run():
        return wrapper.call()


def setup_env(sops_binary="/go/bin/sops"):
    import imprint as ip

    ip.package_settings()

    decrypt_secrets(sops_binary=sops_binary)
    env = dotenv.dotenv_values(None)
    logger.debug("Environment variables loaded: %s", list(env.keys()))
    dotenv.load_dotenv(None)


def decrypt_secrets(sops_binary="/go/bin/sops"):
    if not os.path.exists(".env"):
        p = subprocess.run(
            [sops_binary, "-d", "--output", "./.env", "./test_secrets.enc.env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        logger.debug(
            "Decrypting secrets. stdout from sops:\n %s", p.stdout.decode("utf-8")
        )


def coiled_login():
    token = os.environ["COILED_TOKEN"]
    print(subprocess.check_output(["coiled", "login", "-t", token]).decode("utf-8"))
