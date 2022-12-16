# Confirmasaurus

- [See here for ongoing work.](https://github.com/orgs/Confirm-Solutions/projects/1/views/1)
- [And here for higher level strategy and planning](https://docs.google.com/document/d/1XhVMvYwWAb-27SIsYVME5pClfd-cnSDZxQh-3xJrrJ8/edit)

Directories:

- `imprint` - our open source tools. [Open source repo](https://github.com/Confirm-Solutions/imprint)
- `confirm` - our code base!
- `tests` - our tests.
- `docs` - documentation.
- `research` - Research! Stuff in here might be in various of states of rough/polished.
- `cloud` - [tools for working on cloud infrastructure including AWS and Codespaces.](cloud/README.md)
- Doing something new? Just make a new top level folder.

## Getting set up

See here for a local set up. Look at the [cloud documentation for setting up on AWS or GitHub Codespaces](cloud/README.md).

To get a fully functional development environment...

1. If you do not have conda installed already, please install it. There are
   many ways to get conda. We recommend installing `Mambaforge` which is a
   conda installation wwith `mamba` installed by default and set to use
   `conda-forge` as the default set of package repositories. [CLICK HERE for
   installers and installation
   instructions.](https://github.com/conda-forge/miniforge#mambaforge)
2. Clone the git repo:

   ```bash
   git clone git@github.com:Confirm-Solutions/confirmasaurus.git
   ```

3. Set up your confirm conda environment. The list of packages that will be
   installed inside your conda environment can be seen
   in [`pyproject.toml`](pyproject.toml).

   ```bash
   cd confirm/env
   mamba update -y conda
   # create a development virtual environment with useful tools
   mamba env create
   conda activate confirm
   # install the confirm package plus development tools
   poetry install --with=dev,test,cloud,cloud_dev
   ```
   
## Committing code

In order to commit code and pass the pre-commit checks, you will need to install `gitleaks` with `brew install gitleaks`.


## Other useful notes:

- [Packaging and dependency management tools](./docs/packaging.md)
- [Using our tools on the cloud (AWS, Codespaces)](./cloud/README.md)
- [Advice for dealing with our imprint subtree](./docs/git_subtree.md)
- [Very rough coding standards](./docs/standards.md)
- [Advice for using JAX](./docs/jax_patterns.md)