#!/bin/bash
# This script sets up a terraform-launched instance as a remote docker context.
# This means that all docker commands will be run on the remote instance.

docker context use default
docker context rm remotedev || true

URL=$(terraform output -raw ec2instance)
# alternative command: aws ec2 describe-instances | jq -r ".Reservations[0].Instances[0].PublicDnsName"
USER="ec2-user" # for use with amazon linux

docker context create remotedev \
    --docker host="ssh://$USER@$URL" 
docker context use remotedev

docker login ghcr.io -u "$GITHUB_USER" -p "$GITHUB_TOKEN"
docker pull ghcr.io/confirm-solutions/smalldev