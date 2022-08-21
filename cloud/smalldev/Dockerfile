FROM condaforge/mambaforge as mambaforge_upstream

FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

COPY --from=mambaforge_upstream /opt /opt/

# Copy library scripts to execute
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

COPY environment*.yml /tmp/conda-tmp/
# Set up our conda environment! Since we're inside a container, we'll just
# install everything into the "base" environment.
#
# BUG: the install here does not work correctly when launched inside docker
# from an M1 Mac because it's running Linux on an "aarch64" architecture and
# the PyPI packages from JAX are not properly set up.
#
RUN mamba env update -n base -f /tmp/conda-tmp/environment.yml \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes \
    && pip cache purge
