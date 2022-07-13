#!/bin/bash

# TODO: could also set up conda environment here? or is that too magic?
# List the internal packages that should be available here.
for PKG in ./inlaw ./research/berry
do
    [ -d "$PKG" ] && echo "Directory $PKG exists" && pip install -e "$PKG"
done