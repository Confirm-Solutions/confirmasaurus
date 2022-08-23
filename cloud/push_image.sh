#!/bin/bash

# For the most part, our images should be pushed through Github actions, but if
# you need to push an image manually to ghcr.io, this script gives a template
# for how to do that.
echo "$GITHUB_TOKEN" | docker login ghcr.io -u tbenthompson --password-stdin
docker tag smalldev ghcr.io/confirm-solutions/smalldev:latest
docker push ghcr.io/confirm-solutions/smalldev:latest
