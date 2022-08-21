#!/bin/bash
ACCOUNT=$(aws sts get-caller-identity | jq -r .UserId)
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin "$ACCOUNT".dkr.ecr.us-east-1.amazonaws.com
docker tag smalldev:latest "$ACCOUNT".dkr.ecr.us-east-1.amazonaws.com/smalldev:latest
docker push "$ACCOUNT".dkr.ecr.us-east-1.amazonaws.com/smalldev:latest
