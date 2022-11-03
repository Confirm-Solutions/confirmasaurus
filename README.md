# Confirmasaurus

- [See here for ongoing work.](https://github.com/orgs/Confirm-Solutions/projects/1/views/1)
- [And here for higher level strategy and planning](https://docs.google.com/document/d/1XhVMvYwWAb-27SIsYVME5pClfd-cnSDZxQh-3xJrrJ8/edit)

Directories:

- `imprint` - our open source tools. [Open source repo](https://github.com/Confirm-Solutions/imprint)
- `outlaw` - our fast INLA tools.
- `research` - Research! Stuff in here might be in various of states of rough/polished.
- `anarchy` - [Do whatever you want here.](anarchy/README.md)
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

3. Set up your confirm conda environment. The list of
   packages that will be installed inside your conda environment can be seen in
   the [`environment-dev.yml` file](environment-dev.yml) (for development
   dependencies) and in [`pyproject.toml`](pyproject.toml) (for critical
   dependencies).

   ```bash
   cd confirm/env
   mamba update -y conda
   # create a development virtual environment with useful tools
   mamba env create -f environment-dev.yml
   conda activate confirm
   # install the confirm package
   poetry install
   ```

## Other useful notes:

- [Using our tools on the cloud (AWS, Codespaces)](./cloud/README.md)
- [Advice for dealing with our imprint subtree](./docs/git_subtree.md)
- [Very rough coding standards](./docs/standards.md)

## A list of useful notebook:

It would be good to organize a little bit.

- [Imprint tutorial](./imprint/research/berry/tutorial.ipynb)
- [Intro to INLA](./research/berry/intro_to_inla.ipynb)
- [Berry 2013 Imprint example](./research/berry/berry_imprint.ipynb)
- [Working through getting the conditional full Laplace INLA working](./anarchy/conditional_inla.ipynb)
- [Global optimization of the type I error](./anarchy/opt/opt.ipynb)
- [Comparing against R-INLA](./research/outlaw_vs_rinla/outlaw.ipynb)
- Messy survival analysis notebooks in research/survival replicating stuff in the Thall 2003 and Thall 2005 papers.
