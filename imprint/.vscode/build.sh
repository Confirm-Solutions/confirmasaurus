#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate imprint
bazel build -c opt --config gcc //python:pyimprint/core.so
cp -f ./bazel-bin/python/pyimprint/core.so python/pyimprint/