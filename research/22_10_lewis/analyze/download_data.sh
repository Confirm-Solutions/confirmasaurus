#!/bin/bash

# directory where current shell script resides
PROJECTDIR=$(dirname "$BASH_SOURCE")
cd "$PROJECTDIR"
cd ..
mkdir -p data
cd data
aws s3 cp s3://imprint-dump/output_lei4d/ output_1/ --recursive
aws s3 cp s3://imprint-dump/output_lei4d2/ output_2/ --recursive