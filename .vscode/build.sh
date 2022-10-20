#!/bin/bash
eval "$(conda shell.bash hook)"
cd imprint
sh ./.vscode/build.sh
cd ..