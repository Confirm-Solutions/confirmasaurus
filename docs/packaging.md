# Packaging and dependency management

## Background

Despite being fairly small, we have a somewhat complicated dependency situation due to the variety of different situations we want to support:

1. MacOS + CPU - our main development environment.
2. Linux + CUDA - our main production environment.
3. Linux + CPU - continuous integration, etc.
4. Testing dependencies (e.g. pytest, pre-commit) beyond the minimal installation necessary.
5. Extra development dependencies (e.g. jupyter)

- _Why not install all the development dependencies everywhere?_ It's slow! CI test runtime is down to 1.5 minutes when we only install the minimal necessary dependencies. Saving instance start-up time in cloud batch jobs saves annoyance and money!
- [_Why is python package management so bad??_](https://bbc.github.io/cloudfit-public-docs/packaging/this_way_up.html)

## What are our goals?

1. Reproducible environments --> dependency locking. We want to be able to precisely reproduce imprint output far into the future.
2. Cross-platform dependency management - MacOS + Linux, CUDA + CPU. Jax + CUDA are annoying here.
3. Automatic dependency updates. Dependabot and Renovatebot scan your repository for out-of-date pinned packages and submit automatic PRs updating those packages.
4. Easy and fast installs. Mamba, pip and poetry are all fast enough.

## What we do most places

1. We use conda/mamba to install python and manage virtual environments. See [`environment.yml`](../environment.yml).
2. We use poetry to install critical dependencies: numpy, scipy, jax, etc. See [`pyproject.toml`](../pyproject.toml) and [`poetry.lock`](../poetry.lock).
3. If necessary, we use conda/mamba to install development tools that are more complex and benefit from an OS-level package manager.
4. Docker provides an extra layer of locking and reproducibility, but we are currently not storing images for posterity.

Only step #2 is absolutely necessary.

## Why not use only conda?

1. There's no good solution for dependency locking.
2. Our most challenging package is jax and it is not a conda package!
3. It doesn't help with the CUDA problem at all!
4. Conda isn't supported by dependabot or renovatebot or other dependency updaters.

## Why not use only poetry?

Conda and conda-forge are fantastic for installing lots of python-related packages that might also have non-Python parts or dependencies. This is especially useful for rapid development and "data science".

We're close to only needing poetry and in some situations, we don't need conda. The [CI workflow](../.github/workflows/test.yml) doesn't use conda.

However, without conda, we would:

1. Need an alternative way to install Python. This isn't a hard problem but conda/miniconda/mambaforge are very easy and I like them.
2. Lose access to the large repository of easy-to-install packages that are often not-at-all-easy-to-install with pip/poetry. (e.g. sage!) This is not important for production jobs but it's nice to have this access for development.
3. Need to spend time updating various parts of our cloud setup, mainly the Dockerfiles. Updating Dockerfiles is painful because the iteration time is so slow.

## How to update packages in poetry.lock and pyproject.toml?

- `poetry self update` --> update `poetry` itself.
- `poetry lock` --> update `poetry.lock` based on `pyproject.toml`
- `poetry update` --> update package versions according to `pyproject.toml`
- `poetry up --with=test,cloud,dev,cloud_dev` --> this will update the package versions in `pyproject.toml`
- [The poetry plugin "poetry up" helps here.](https://github.com/MousaZeidBaker/poetry-plugin-up)
- [Lots of discussion in this issue](https://github.com/python-poetry/poetry/issues/461)

## Notes

- [JAX doesn't have a "normal" package registry and doesn't use PyPI](https://github.com/google/jax/issues/5410)
- [Some discussion on using conda and poetry together](https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry)
- [Useful thoughts on different dependency managers](https://www.recursion.com/news/recursion-developer-experience-evaluating-python-dependency-managers)
- conda and `conda env export` solves some of the dependency locking problem but has a lot of
- conda-lock - tries to solve the package locking issue for conda. Doesn't work well for us because our annoying dependency is jax which is not conda: https://pythonspeed.com/articles/conda-dependency-management/
- environment.yml doesn't support platform selectors, but it is one of the most requested features. https://github.com/conda/conda/issues/8089 https://stackoverflow.com/questions/32869042/is-there-a-way-to-have-platform-specific-dependencies-in-environment-yml
- pip-tools (pip-compile) - this partially solves the narrow issue of locking package versions but doesn't do it as well as poetry and doesn't have the broader package building/distribution benefits of poetry.
- `poetry config virtualenvs.create false` is useful in CI an Docker to use the base conda environment.
