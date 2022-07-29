# Most of what is here is modified from:
# https://github.com/microsoft/vscode-dev-containers/blob/v0.238.1/containers/python-3-miniconda/.devcontainer/base.Dockerfile
FROM condaforge/mambaforge as upstream

# Update, change owner
RUN groupadd -r conda --gid 900 \
    && chown -R :conda /opt/conda \
    && chmod -R g+w /opt/conda \
    && find /opt -type d | xargs -n 1 chmod g+s
    
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

COPY --from=upstream /opt /opt/

# Copy library scripts to execute
COPY .devcontainer/library-scripts/*.sh .devcontainer/add-notice.sh .devcontainer/library-scripts/*.env /tmp/library-scripts/
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
    && bash /tmp/library-scripts/add-notice.sh \
    && mv -f "/tmp/library-scripts/meta.env" /usr/local/etc/vscode-dev-containers/meta.env \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && groupadd -r conda --gid 900 \
    && usermod -aG conda ${USERNAME} \
    && /tmp/library-scripts/add-notice.sh

# Install R and enough tools to run R scripts and test r-inla
# This R install was copied from: https://github.com/rocker-org/rocker/blob/master/r-ubuntu/jammy/Dockerfile
# and combined with the advice from here: https://cran.r-project.org/bin/linux/ubuntu/
# I also used the instructions here: 
# - https://github.com/REditorSupport/vscode-R/wiki/Installation:-Linux
# - https://code.visualstudio.com/docs/languages/r
# the advice here is also useful: https://schiff.co.nz/blog/r-and-vscode/
# NOTE: this could be easily split out as part of a multistage build
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y --no-install-recommends \
        software-properties-common dirmngr \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
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
    && mamba install -c conda-forge radian \
  	&& ln -s /usr/lib/R/site-library/littler/examples/install.r /usr/local/bin/install.r \
    && install.r --error \
        languageserver \
        httpgd \
    && R -e "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=FALSE)" \
 	&& rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
 	&& rm -rf /var/lib/apt/lists/*

# Install Node!
# Node is used for our frontend tool, but is also handy since there are a lot
# of useful small packages built on node
ENV NODE_VERSION="v18"
ENV NVM_DIR=/usr/local/share/nvm
RUN mkdir -p $NVM_DIR \
    && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install ${NODE_VERSION} \
    && nvm use ${NODE_VERSION} \
    && nvm alias default ${NODE_VERSION} \
    && echo -e "export PATH=$(dirname $(nvm which v18)):$PATH" >> /etc/bash.bashrc \
    && echo -e "export PATH=$(dirname $(nvm which v18)):$PATH" >> /etc/zsh/zshrc

# Set up our conda environment! Since we're inside a container, we'll just
# install everything into the "base" environment.
# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# After the env update, 
COPY environment*.yml .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
    umask 0002 \
    && /opt/conda/bin/mamba env update -n base -f /tmp/conda-tmp/environment.yml \
    && /opt/conda/bin/mamba env update -n base -f /tmp/conda-tmp/environment-dev.yml; \
    fi \
    && rm -rf /tmp/conda-tmp
    
# Now that Jupyter is installed, we insert the R kernel.
RUN R -e "IRkernel::installspec()"

# Finally, install neovim and latex since these are commonly needed tools.
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends\
        neovim\
        texlive-latex-extra\
    && apt-get upgrade -y\
    && apt-get clean -y\
    && rm -rf /var/lib/apt/lists/*

# RUN usermod -s /bin/zsh root