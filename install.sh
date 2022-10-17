#!/bin/bash

if ! [ -x "$(command -v nvidia-smi)" ] && [[ -z "$CONFIRM_FORCE_JAX_CUDA" ]]; then
    echo 'No GPUs available. Replacing jax[cuda] with jax[cpu].' >&2
    pip uninstall -y jax jaxlib numpyro
    pip install "jax[cpu]" numpyro
fi

# Set up pre-commit so it's fast the first time it gets used
pre-commit install --install-hooks

pushd imprint || exit
./generate_bazelrc
popd || exit

# List the internal packages that should be available here.
for PKG in ./confirm ./imprint/python
do
    [ -d "$PKG" ] && echo -e "\nDirectory $PKG exists. Installing... \n" && pip install --no-deps -e "$PKG"
done

# Set up our imprint remote so we can use subtree. On Codespaces, we need
# https. Locally, we use ssh.
if [[ -n "$CONFIRM_IMPRINT_HTTPS" ]]; then
    git remote add -f imprint https://github.com/Confirm-Solutions/imprint.git
fi

if [[ -n "$CONFIRM_IMPRINT_SSH" ]]; then
    git remote add -f imprint git@github.com:Confirm-Solutions/imprint.git
fi
