# A very small Dockerfile that is sufficient to run our main Python/JAX code.
FROM condaforge/mambaforge as mambaforge_upstream

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY --from=mambaforge_upstream /opt /opt/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

COPY environment.yml /tmp/conda-tmp/
RUN mamba env update -n base -f /tmp/conda-tmp/environment.yml \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes \
    && pip cache purge

COPY pyproject.toml poetry.lock ./confirm /confirmasaurus/
RUN cd /confirmasaurus && poetry install --without test
WORKDIR /confirmasaurus