# See the devimage repo here for setting up the image here:
# https://github.com/Confirm-Solutions/devimage
# I think there could be a better way to do this, but this works for now.
# As a general rule:
# - anything that would be installed as part of our README.md file should be installed here.
# - anything that would be assumed to be installed as part of a dev's system should be in the devimage.
FROM docker pull ghcr.io/confirm-solutions/devimage:main

# Set up our conda environment! Since we're inside a container, we'll just
# install everything into the "base" environment.
# NOTE: installing g++ gcc binutils through conda results in conda activate
# being slightly slow. this is weird and kind of a bummer.
# NOTE: most of the things here are already installed by the devimage. If you
# add new dependencies, adding them to the devimage will make the image build
# faster
COPY environment*.yml /tmp/conda-tmp/
RUN mamba env update -n base -f /tmp/conda-tmp/environment.yml \
    && mamba env update -n base -f /tmp/conda-tmp/environment-dev.yml \
    && rm -rf /tmp/conda-tmp \
    && mamba clean --all --yes