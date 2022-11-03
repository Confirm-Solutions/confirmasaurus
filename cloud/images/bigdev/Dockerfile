# Many things here are modified from:
# https://github.com/microsoft/vscode-dev-containers/blob/v0.238.1/containers/python-3-miniconda/.devcontainer/base.Dockerfile
# mambaforge image: https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
# This image kind of sucks to build. It takes around 15 minutes because we're installing a lot of stuff:
# - lots of apt packages (1.5 minutes)
# - the sage installation (~8 minutes)
# - cuda (3 minutes)
# - R packages (5 minutes)
# - the whole environment*.yml conda environment (3 minutes)
# - the Node install is actually really fast.
# There are lots of ways to solve this, but they all require effort and I don't
# think it's worth it right now.
FROM condaforge/mambaforge as mambaforge_upstream

# The devcontainer base image is quite useful and sets up a lots of stuff for us. 
# It would be nice to be able to use a larger base image, but that seems like
# it would require building the base image ourselves, which is a chore and
# added maintenance burden.
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

COPY --from=mambaforge_upstream /opt /opt/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"

# Install useful basic tools... boring stuff, but also neovim and latex
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
        # Install neovim and texlive since they're useful!
        neovim \
        texlive-latex-extra \
        cm-super \
        dvipng \
    # I don't know why this line is here, but I don't want to debug any
    # problems that might crop up from deleting it.
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    # Install conda into root's bash!
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
# Install R and enough tools to run R scripts and test r-inla
# This R install was copied from: https://github.com/rocker-org/rocker/blob/master/r-ubuntu/jammy/Dockerfile
# and combined with the advice from here: https://cran.r-project.org/bin/linux/ubuntu/
# I also used the instructions here: 
# - https://github.com/REditorSupport/vscode-R/wiki/Installation:-Linux
# - https://code.visualstudio.com/docs/languages/r
# the advice here is also useful: https://schiff.co.nz/blog/r-and-vscode/
# NOTE: this could be easily split out as part of a multistage build
# NOTE: radian repl is install throughed environment-dev.yml
# NOTE: it is possible to install R through conda instead. I tried this, but it
# adds a bunch of code to conda activate that results in conda activate running
# SHOCKINGLY slowly: https://github.com/conda/conda/issues/8806
# RUN \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
    && apt-get update \
    # Install any R packages that we can from apt because these are binary
    # packages whereas using install.packages or install2.r will install from
    # source and require a build.
    # The apt repo here gets us the latest version of R.
    && apt-get install -y --no-install-recommends \
        littler \
        r-base \
        r-base-dev \
        r-recommended \
        r-cran-docopt \
        r-cran-roxygen2 \
        r-cran-rmarkdown \
        r-cran-pbdzmq \
        r-cran-uuid \
        r-cran-jsonlite \
        r-cran-irkernel \
        r-cran-foreach \
        r-cran-sp \
        r-cran-ggplot2 \
        libcairo2-dev \
        libfontconfig1-dev \
  	&& ln -s /usr/lib/R/site-library/littler/examples/install2.r /usr/local/bin/install2.r \
    # this step will install several packages that look like they're already
    # installed. this is because languageserver depends on a different (newer?)
    # version than in the r-cran apt repo
    && install2.r --skipinstalled --ncpus 6 --error \
        languageserver \
        httpgd \
    # from https://www.r-inla.org/download-install
    # we set dep=FALSE because INLA pulls in the kitchen sink but doesn't
    # actually need 95% of its dependencies for our main use cases.
    && R -e "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=FALSE)" \
    # clean up to shrink image size a little bit.
 	&& rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && apt-get clean -y \
    && apt-get autoremove -y


# Based on the Dockerfiles here:
# https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist/11.7.0/ubuntu2204
# We combine the packages from the base, runtime and devel images.
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


# Install Node!
# Node is used for our frontend tool, but is also handy since there are a lot
# of useful small packages built on node
# The NODE_VERSION should ideally be the latest LTS release.
ENV NODE_VERSION="v18"
ENV NVM_DIR=/usr/local/share/nvm
RUN mkdir -p $NVM_DIR \
    && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install ${NODE_VERSION} \
    && nvm use ${NODE_VERSION} \
    && nvm alias default ${NODE_VERSION} \
    # why do we export PATH like this instead of using sourcing nvm.sh like normal?
    # 1. we only need a single node version. we're not doing complex multi-version crazy stuff
    # 2. adding nvm.sh to my shell was making it super slow!!
    && echo "export PATH=$(dirname $(nvm which v18)):\$PATH" >> /etc/bash.bashrc \
    && echo "export PATH=$(dirname $(nvm which v18)):\$PATH" >> /etc/zsh/zshrc

# Create a sagemath conda environment.
# Since the environment is part of our conda installation, the resulting
# jupyter kernel can be selected from VSCode's kernel selector.
COPY environment-dev.yml pyproject.toml poetry.lock /tmp/conda-tmp/
RUN mamba create -n sagemath -y sage \
# Set up our conda environment! Since we're inside a container, we'll just
# install everything into the "base" environment.
#
# BUG: the install here does not work correctly when launched inside docker
# from an M1 Mac because it's running Linux on an "aarch64" architecture and
# the PyPI packages from JAX are not properly set up.
#
# NOTE: installing g++ gcc binutils through conda results in conda activate
# being slightly slow. this is weird and kind of a bummer.
    && mamba env update -n base -f /tmp/conda-tmp/environment-dev.yml \
    && poetry install \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes \
    && pip cache purge \
    # Now that Jupyter is installed, we install the R kernel from IRkernel. This
    # means you can run R in Jupyter!
    && R -e "IRkernel::installspec()"

# Install AWS CLI V2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
  && unzip awscliv2.zip \
  && sudo ./aws/install

LABEL org.opencontainers.image.source https://github.com/Confirm-Solutions/confirmasaurus