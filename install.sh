#!/bin/bash

# Install our package:
poetry install

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