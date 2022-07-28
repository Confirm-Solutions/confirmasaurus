# COPY
# COPY
# COPY
# Starting here, this is modified from:
# https://github.com/microsoft/vscode-dev-containers/blob/v0.238.1/containers/python-3-miniconda/.devcontainer/base.Dockerfile
FROM continuumio/miniconda3 as upstream

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
    && apt-get upgrade -y \
    && bash /tmp/library-scripts/add-notice.sh \
    && mv -f "/tmp/library-scripts/meta.env" /usr/local/etc/vscode-dev-containers/meta.env \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && groupadd -r conda --gid 900 \
    && usermod -aG conda ${USERNAME} \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/add-notice.sh

# Setup default python tools in a venv via pipx to avoid conflicts
# Ben Note: this is unnecessary
# ENV PIPX_HOME=/usr/local/py-utils \
#     PIPX_BIN_DIR=/usr/local/py-utils/bin
# ENV PATH=${PATH}:${PIPX_BIN_DIR}
# RUN bash /tmp/library-scripts/python-debian.sh "none" "/opt/conda" "${PIPX_HOME}" "${USERNAME}" "true" \ 
#     && apt-get clean -y && rm -rf /var/lib/apt/lists/* .devcontainer/library-scripts/python-debian.sh
    
ARG NODE_VERSION="none"
ENV NVM_DIR=/usr/local/share/nvm
ENV NVM_SYMLINK_CURRENT=true \
    PATH=${NVM_DIR}/current/bin:${PATH}
RUN bash /tmp/library-scripts/node-debian.sh "${NVM_DIR}" "${NODE_VERSION}" "${USERNAME}" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* .devcontainer/library-scripts
# END COPY
# END COPY
# END COPY
    
    
# A little funny that the section above looks like it might install node. But
# it seems to fail to do so even if I set NODE_VERSION. So, this is the actual
# node installation.
ARG NODE_VERSION="lts/*"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Install mamba for faster environment install
RUN /opt/conda/bin/conda install -y -c conda-forge mamba
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

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends\
    neovim\
    texlive-latex-extra