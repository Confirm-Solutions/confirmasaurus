#!/bin/bash

# AWS ECR instructions:

# ACCOUNT=$(aws sts get-caller-identity | jq -r .UserId)
# aws ecr get-login-password --region us-east-1 | \
#     docker login --username AWS --password-stdin "$ACCOUNT".dkr.ecr.us-east-1.amazonaws.com
# docker tag smalldev:latest "$ACCOUNT".dkr.ecr.us-east-1.amazonaws.com/smalldev:latest
# docker push "$ACCOUNT".dkr.ecr.us-east-1.amazonaws.com/smalldev:latest

# Github ghcr instructions:

echo $GITHUB_TOKEN | docker login ghcr.io -u tbenthompson --password-stdin
docker tag smalldev ghcr.io/Confirm-Solutions/smalldev:latest
docker push ghcr.io/confirm-solutions/smalldev:latest
