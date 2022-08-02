# Confirmasaurus

- [See here for ongoing work.](https://github.com/orgs/Confirm-Solutions/projects/1/views/1)
- [And here for higher level strategy and planning](https://docs.google.com/document/d/1XhVMvYwWAb-27SIsYVME5pClfd-cnSDZxQh-3xJrrJ8/edit)

Here:
- `imprint` - our open source tools.
- `outlaw` - our fast INLA tools
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
    mamba update -y conda
    mamba env create
    conda activate confirm
    # This final line is only necessary if you want "extra" tools that are
    # helpful for development but shouldn't be dependencies of any of our main
    # code.
    conda env update -n confirm -f environment-dev.yml
    ```
4. Get set up with pre-commit and our internal python libraries by running:

    ```bash
    CONFIRM_IMPRINT_SSH=1 ./install.sh
    ```


## Using GitHub Codespaces with the confirmasaurus repo

To start using Codespaces, follow along with the [quick start](https://docs.github.com/en/codespaces/getting-started/quickstart) but just use our repo instead of the example repo. For the moment, I strongly recommend using the GPU configuration since it's free while the beta program is still ongoing. Also, they're really fast! 6 cores, 112G RAM, 1 NVidia V100 GPU.

The dev environment should be set up quite cleanly. To see some of the relevant configuration:
- [devcontainer.json](.devcontainer/devcontainer.json)
- [Dockerfile](.devcontainer/Dockerfile)
- the [environment.yml](environment.yml) and [environment-dev.yml](environment-dev.yml) are used to define the conda environment.

The set up focuses on things that anyone using our repo should have installed and set up. It sets up a conda environment including all our standard tools. It gives a good development experience for:
- Python: VSCode extensions, all our dependencies, linting, formatting, debugging, pytest.
- Jupyter: lovely.
- C++: linting, formatting (using clang format), compilation and debugging integrated into the IDE.
- R: handles R markdown documents, [also Jupyter with R](https://stackoverflow.com/a/67320274/3817027), r-inla, debugging, etc. Actually a pretty nice R development experience. 
- Latex and markdown and a lot more.

After you've set up your codespace, all the steps above in "Getting set up" above will have already been done for you! There's no setup left to do. Just code and enjoy.

For personal customizations, you get to do two things. These are not mutually exclusive and allow configuring different components of your environment.
- you can either run the browser-based VSCode that is the Codespaces default or you can connect to the remote machine from your local VSCode app. I much prefer connecting with my local app but I haven't tried the browser version very much! 
- you can use [VSCode Settings Sync](https://docs.github.com/en/codespaces/customizing-your-codespace/personalizing-github-codespaces-for-your-account#settings-sync) (you sign in to your github account) to sync settings VSCode settings, keybindings and extensions between local machine and codespace. This is nice and I strongly recommend it!!
- you can create a [dotfiles repo](https://docs.github.com/en/codespaces/customizing-your-codespace/personalizing-github-codespaces-for-your-account#dotfiles) that the codespace will clone and copy in automatically. These will set up your terminal development environment including things like your bash/zsh configuration, any tools you would like to install with `apt` and any other configuration/installation that you want to automate. I have everything set up so that the environment/tools are essentially indistinguishable from my local machine. This is even nicer and I really strongly recommend this! (Note: take a look at the [Dockerfile](.devcontainer/Dockerfile) to see what tools are already installed.)

I am happy to share my dotfiles and VSCode settings if you'd like. To share the dotfiles, I'll need to scrub out some passwords first, but that's probably an improvement anyway. :embarrassed:

## Developing the docker image

- Install docker locally or launch a codespace with the "docker" configuration.
- Run the `build` script in `.devcontainer/bigdev`. Look at the script. It's just a one liner.
- The `test` script checks that everything is installed properly.
- Once you commit changes to the `.devcontainer/bigdev/Dockerfile` to the `main` branch, Codespace
  prebuilds will be generated by an actions workflow. Try to work on the docker
  image in a branch so you don't pummel our actions budget too much.
## Other useful notes:

- [Advice for dealing with our imprint subtree](./docs/GitSubtree.md)
- [Very rough coding standards](./docs/standards.md)