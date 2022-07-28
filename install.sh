#!/bin/bash

# List the internal packages that should be available here.
pre-commit install --install-hooks

for PKG in ./inlaw ./research/berry ./imprint/python
do
    [ -d "$PKG" ] && printf '\n Directory $PKG exists' && pip install --no-deps -e "$PKG"
done