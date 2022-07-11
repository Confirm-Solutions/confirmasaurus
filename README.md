# Confirmasaurus

- `imprint` - our open source tools.
- `research` - Research! Stuff in here might be in various of states of rough/polished.
- `anarchy` - [Do whatever you want here](anarchy/README.md)
- Doing something new? Just make a new top level folder.

## Getting set up

Please run all the steps here to get a fully functional development environment.

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
3. Set up your confirm conda environment (note that you may substitute `mamba`
   here for `conda` and the install will be substantially faster). The list of
   packages that will be installed inside your conda environment can be seen in
   the [`environment.yml` file](environment.yml).

    ```bash
    cd confirm/
    conda update -y conda
    conda env create
    conda activate confirm
    ```
4. To set up pre-commit for this git repo:

    ```bash
    pre-commit install
    ```
5. Get set up with our internal python libraries by running:

    ```bash
    ./install.sh
    ```
