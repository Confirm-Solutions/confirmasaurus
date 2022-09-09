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

RUN mamba install --yes --freeze-installed \
        numpy scipy matplotlib pip \
    && pip install \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
          "jax[cuda]" \
          "numpyro[cuda]" \
    && mamba clean --all --yes \
    && pip cache purge

COPY ./confirm /confirmasaurus/confirm
RUN pip install -e /confirmasaurus/confirm
WORKDIR /confirmasaurus