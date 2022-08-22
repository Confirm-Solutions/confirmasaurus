#!/bin/bash

# Connects to the terraform 

URL=$(terraform output -raw ec2instance)
# alternative command: aws ec2 describe-instances | jq -r ".Reservations[0].Instances[0].PublicDnsName"
USER="ec2-user" # for use with amazon linux
# USER="ubuntu" # for use with ubuntu

echo -e "\n Adding AWS key to ssh-agent. \n"
eval `ssh-agent -s`
ssh-add - <<< "${AWS_KEY_AWS_KEY_PAIR}"
ssh-add - <<< "${AWS_GITHUB_KEY}"

echo -e "\n Connecting to: $USER@$URL \n"
ssh -A "$USER@$URL" 


# docker run -it --gpus all --network host -v `pwd`:/workspaces/confirmasaurus smalldev ls /workspaces/confirmasaurus