#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate imprint
bazel build //python:pyimprint/core.so
cp ./bazel-bin/python/pyimprint/core.so python/pyimprint/core.so