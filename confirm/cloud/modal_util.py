from pathlib import Path

import modal


def get_image():
    poetry_dir = Path(__file__).resolve().parent.parent.parent

    context_files = {
        "/.pyproject.toml": str(poetry_dir.joinpath("pyproject.toml")),
        "/.poetry.lock": str(poetry_dir.joinpath("poetry.lock")),
    }

    dockerfile_commands = [
        "RUN python -m pip install poetry",
        "COPY /.poetry.lock /tmp/poetry/poetry.lock",
        "COPY /.pyproject.toml /tmp/poetry/pyproject.toml",
        "RUN cd /tmp/poetry && \\ ",
        "  poetry config virtualenvs.create false && \\ ",
        "  poetry install --with=cloud --no-root",
    ]

    return modal.Image.from_dockerhub(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        setup_commands=[
            "apt-get update",
            "apt-get install -y python-is-python3 python3-pip",
        ],
    ).dockerfile_commands(dockerfile_commands, context_files=context_files)
