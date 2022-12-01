FROM condaforge/mambaforge as mambaforge_upstream

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY --from=mambaforge_upstream /opt /opt/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
    
ARG USERNAME="root"

COPY environment.yml /
RUN mamba env update -n base -f /environment.yml \
    && mamba clean --all --yes

COPY pyproject.toml poetry.lock /
RUN poetry config virtualenvs.create false \
    && poetry install --no-root \
    && cd / \
    && rm -rf /tmp/conda-tmp \
    && pip cache purge
    
COPY main.py /
ENTRYPOINT ["python", "/main.py"]

# FROM tensorflow/tensorflow:latest-gpu

# RUN pip install -U pip && \
#     pip install poetry && \
#     pip cache purge

# COPY main.py pyproject.toml poetry.lock /
# RUN cd / && \
#     poetry config virtualenvs.create false && \
#     poetry install --no-root && \
#     pip cache purge

# ENTRYPOINT ["python", "/main.py"]
#
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# RUN apt-get update -y && \
#     apt-get install -y python-is-python3 python3-pip \
#     && apt-get clean -y \
#     && apt-get autoremove -y

# COPY main.py pyproject.toml poetry.lock /
# RUN pip install -U pip && \
#     pip install poetry && \
#     cd / && \
#     poetry install --no-root && \
#     pip cache purge

# ENTRYPOINT ["python", "/main.py"]