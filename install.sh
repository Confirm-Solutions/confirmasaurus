#!/bin/bash

# We don't use poetry envs. The caller should have already activated a conda
# environment.
poetry config virtualenvs.create false

# Install dependencies. This might fail in Codespaces or Dev Containers due to
# not being run as root. That's okay because in those settings, we've already
# installed our dependencies in the Dockerfile.
poetry install || true

# Install our package:
poetry install --only-root

# Set up pre-commit so it's fast the first time it gets used
pre-commit install --install-hooks

# Set up our imprint remote so we can use subtree. On Codespaces, we need
# https. Locally, we use ssh.
if [[ -n "$CONFIRM_IMPRINT_HTTPS" ]]; then
    git remote add -f imprint https://github.com/Confirm-Solutions/imprint.git
fi

if [[ -n "$CONFIRM_IMPRINT_SSH" ]]; then
    git remote add -f imprint git@github.com:Confirm-Solutions/imprint.git
fi