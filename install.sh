#!/bin/bash

# List the internal packages that should be available here.
for PKG in ./inlaw ./research/berry ./imprint/python
do
    [ -d "$PKG" ] && echo "\nDirectory $PKG exists" && pip install --no-deps -e "$PKG"
done