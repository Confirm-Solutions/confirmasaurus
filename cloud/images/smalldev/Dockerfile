# This Dockerfile is a subset of the one in .devcontainer/bigdev/Dockerfile.
# Look there for details and commentary.

FROM condaforge/mambaforge as mambaforge_upstream

FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

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

ARG CUDA_PKG_VERSION="11-7"
ENV CUDA_VERSION 11.7.0
ENV NVARCH=x86_64
RUN curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt update \
    && apt install -y --no-install-recommends \
        cuda-cudart-${CUDA_PKG_VERSION} \
        cuda-compat-${CUDA_PKG_VERSION} \
        cuda-libraries-${CUDA_PKG_VERSION} \
        libnpp-${CUDA_PKG_VERSION} \
        cuda-nvtx-${CUDA_PKG_VERSION} \
        libcusparse-${CUDA_PKG_VERSION} \
        libcublas-${CUDA_PKG_VERSION} \
        cuda-cudart-dev-${CUDA_PKG_VERSION} \
        cuda-command-line-tools-${CUDA_PKG_VERSION} \
        cuda-minimal-build-${CUDA_PKG_VERSION} \
        cuda-libraries-dev-${CUDA_PKG_VERSION} \
        cuda-nvml-dev-${CUDA_PKG_VERSION} \
        cuda-nvprof-${CUDA_PKG_VERSION} \
        libnpp-dev-${CUDA_PKG_VERSION} \
        libcusparse-dev-${CUDA_PKG_VERSION} \
        libcublas-dev-${CUDA_PKG_VERSION} \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

COPY environment.yml pyproject.toml poetry.lock /tmp/conda-tmp/
RUN cd /tmp/conda-tmp/ \
    && mamba env update -n base -f environment.yml \
    && poetry config virtualenvs.create false \
    && poetry install \
    && cd / \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes \
    && pip cache purge
    
# Install AWS CLI V2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
  && unzip awscliv2.zip \
  && sudo ./aws/install

LABEL org.opencontainers.image.source https://github.com/Confirm-Solutions/confirmasaurus