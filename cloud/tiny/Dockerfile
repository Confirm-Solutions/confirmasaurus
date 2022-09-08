# A very small Dockerfile that is sufficient to run our main Python/JAX code.
FROM condaforge/mambaforge as mambaforge_upstream

FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

COPY --from=mambaforge_upstream /opt /opt/

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

COPY cloud/tiny/environment-minimal.yml /tmp/conda-tmp/
RUN mamba env update -n base -f /tmp/conda-tmp/environment-minimal.yml \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes \
    && pip cache purge

