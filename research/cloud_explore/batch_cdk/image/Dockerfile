FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y python-is-python3 python3-pip \
    && apt-get clean -y \
    && apt-get autoremove -y

COPY main.py pyproject.toml poetry.lock /
RUN pip install -U pip && \
    cd /tmp/poetry && \
    poetry install --no-root

ENTRYPOINT ["python", "/main.py"]