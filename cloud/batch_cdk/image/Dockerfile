FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y python-is-python3 python3-pip \
    && apt-get clean -y \
    && apt-get autoremove -y

COPY pyproject.toml poetry.lock requirements-jax-cuda.txt /
RUN pip install -U pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --with=cloud --no-root \
    && pip install --upgrade -r requirements-jax-cuda.txt \
    && cd / \
    && rm -rf /tmp/conda-tmp \
    && pip cache purge

COPY main.py /
ENTRYPOINT ["python", "/main.py"]
