#!/bin/bash
eval "$(conda shell.bash hook)"
cd imprint
./.vscode/build.sh
cd ..