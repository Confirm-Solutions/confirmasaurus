# FROM python:3.10-slim-bullseye
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install poetry
COPY poetry2.lock /tmp/poetry/poetry.lock
COPY pyproject2.toml /tmp/poetry/pyproject.toml
RUN python3 -m pip install "jax[cuda11_cudnn82]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html