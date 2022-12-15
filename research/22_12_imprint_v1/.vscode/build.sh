#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate imprint
bazel build -c opt //python:pyimprint/core.so
rm -f python/pyimprint/core.so
cp ./bazel-bin/python/pyimprint/core.so python/pyimprint/