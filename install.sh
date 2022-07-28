#!/bin/bash

# List the internal packages that should be available here.
pre-commit install --install-hooks

for PKG in ./inlaw ./research/berry ./imprint/python
do
    [ -d "$PKG" ] && printf "\nDirectory $PKG exists. Installing... \n" && pip install --no-deps -e "$PKG"
done

git remote add -f imprint git@github.com:Confirm-Solutions/imprint.git