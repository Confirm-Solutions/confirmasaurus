import subprocess
from pathlib import Path

import modal


def get_image(dependency_groups=["cloud"]):
    poetry_dir = Path(__file__).resolve().parent.parent.parent

    context_files = {
        "/.pyproject.toml": str(poetry_dir.joinpath("pyproject.toml")),
        "/.poetry.lock": str(poetry_dir.joinpath("poetry.lock")),
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
        f"""
        RUN cd /tmp/poetry && \\
            poetry config virtualenvs.create false && \\
            poetry install --with={','.join(dependency_groups)} --no-root
        """,
    ]

    return modal.Image.from_dockerhub(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        setup_commands=[
            "apt-get update",
            "apt-get install -y python-is-python3 python3-pip",
        ],
    ).dockerfile_commands(dockerfile_commands, context_files=context_files)


def modalize(stub, **kwargs):
    def decorator(f):
        return stub.function(
            raw_f=f,
            image=get_image(dependency_groups=["test", "cloud"]),
            retries=0,
            mounts=(modal.create_package_mounts(["confirm", "imprint"])),
            secret=modal.Secret.from_name("confirm-secrets"),
            name=f.__qualname__,
            **kwargs,
        )()

    return decorator


def run_on_modal(f, **kwargs):
    stub = modal.Stub("arbitrary_runner")
    wrapper = modalize(stub, **kwargs)(f)
    with stub.run():
        return wrapper.call()


def decrypt_secrets(sops_binary="/go/bin/sops"):
    subprocess.run([sops_binary, "-d", "--output", ".env", "test_secrets.enc.env"])
