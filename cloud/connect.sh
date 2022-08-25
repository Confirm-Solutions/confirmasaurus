#!/bin/bash

# Connects to the instance launched by terraform 
URL=$(terraform output -raw ec2instance)
USER="ec2-user" # for use with amazon linux

echo -e "\n Connecting to: $USER@$URL \n"

# The `-A` flags enables ssh agent forwarding so that your local ssh keys will be
# available on the remote host. This allows you to clone from github, for
# example.
ssh -A "$USER@$URL" 