#!/bin/bash

# Install SOPS and decrypt secrets.
go install go.mozilla.org/sops/cmd/sops@latest
/go/bin/sops -d --output .env test_secrets.enc.env

# We don't use poetry envs. The caller should have already activated a conda
# environment.
poetry config virtualenvs.create false --local

# Install our package (if the above line failed, we still want to make sure we
# install the confirm package itself!)
poetry install --only-root

# Set up pre-commit so it's fast the first time it gets used
pre-commit install --install-hooks

# Set up our imprint remote so we can use subtree. On Codespaces, we need
# https. Locally, we use ssh.
git remote add -f imprint https://github.com/Confirm-Solutions/imprint.git || true