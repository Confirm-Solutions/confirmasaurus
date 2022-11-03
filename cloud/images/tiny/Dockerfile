# A very small Dockerfile that is sufficient to run our main Python/JAX code.
FROM condaforge/mambaforge as mambaforge_upstream

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

COPY --from=mambaforge_upstream /opt /opt/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

COPY environment.yml pyproject.toml poetry.lock /tmp/conda-tmp/
RUN mamba env update -n base -f /tmp/conda-tmp/environment.yml \
    && poetry install \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes \
    && pip cache purge

COPY ./confirm /confirmasaurus/confirm
RUN pip install -e /confirmasaurus/confirm
WORKDIR /confirmasaurus