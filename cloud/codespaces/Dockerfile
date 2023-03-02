# This Dockerfile is a subset of the one in .devcontainer/bigdev/Dockerfile.
# Look there for details and commentary.

FROM condaforge/mambaforge as mambaforge_upstream

FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

COPY --from=mambaforge_upstream /opt /opt/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        software-properties-common \
        dirmngr \
        neovim \
        cm-super \
        dvipng \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
 	&& rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && apt-get clean -y \
    && apt-get autoremove -y

COPY environment.yml requirements.txt requirements-jax-cuda.txt pyproject.toml poetry.lock /tmp/conda-tmp/
RUN cd /tmp/conda-tmp/ \
    && mamba env update -n base -f environment.yml \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --with=test,dev,cloud \
    && pip install --upgrade -r requirements-jax-cuda.txt \
    && cd / \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes \
    && pip cache purge

LABEL org.opencontainers.image.source https://github.com/Confirm-Solutions/confirmasaurus