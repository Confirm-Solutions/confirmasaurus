#!/bin/zsh
eval "$(conda shell.zsh hook)"
cd imprint
./.vscode/build.sh
cd ..