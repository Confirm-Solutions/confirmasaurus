# Many things here are modified from:
# https://github.com/microsoft/vscode-dev-containers/blob/v0.238.1/containers/python-3-miniconda/.devcontainer/base.Dockerfile
# mambaforge image: https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
# This image kind of sucks to build. It takes around 10 minutes because we're installing a lot of stuff:
# - lots of apt packages (1.5 minutes)
# - R packages (5 minutes)
# - the whole environment*.yml conda environment (3 minutes)
# - the Node install is actually really fast.
FROM condaforge/mambaforge as upstream
    
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

COPY --from=upstream /opt /opt/

# Copy library scripts to execute
COPY .devcontainer/library-scripts/*.env /tmp/library-scripts/
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"

# Install useful basic tools... boring stuff, but also neovim and latex
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
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
        texlive-latex-extra\
    # I don't know what this meta.env file does, but all the Codespaces images
    # have it, so I've left it in
    && mv -f "/tmp/library-scripts/meta.env" /usr/local/etc/vscode-dev-containers/meta.env \
    # I don't know why this line is here, but I don't want to debug any
    # problems that might crop up from deleting it.
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    # Install conda into root's bash!
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

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
RUN export DEBIAN_FRONTEND=noninteractive \
    # Install any R packages that we can from apt because these are binary
    # packages whereas using install.packages or install2.r will install from
    # source and require a build.
    # The apt repo here gets us the latest version of R.
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
    && apt-get update \
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
    # actually need 95% of its dependencies.
    && R -e "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=FALSE)" \
    # clean up to shrink image size a little bit.
 	&& rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && apt-get clean -y\
    && rm -rf /var/lib/apt/lists/*

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

# Set up our conda environment! Since we're inside a container, we'll just
# install everything into the "base" environment.
# The instructions here are mostly copied from the miniconda codespaces
# devcontainer dockerfile but I swapped in mamba.
# NOTE: installing g++ gcc binutils through conda results in conda activate
# being slightly slow. this is weird and kind of a bummer.
COPY environment*.yml /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
    umask 0002 \
    && /opt/conda/bin/mamba env update -n base -f /tmp/conda-tmp/environment.yml \
    && /opt/conda/bin/mamba env update -n base -f /tmp/conda-tmp/environment-dev.yml; \
    fi \
    && rm -rf /tmp/conda-tmp
    
# Now that Jupyter is installed, we install the R kernel from IRkernel. This
# means you can run R in Jupyter!
RUN R -e "IRkernel::installspec()"

# I've commented this out because I don't know if we should force everyone to
# use zsh.
# RUN usermod -s /bin/zsh root