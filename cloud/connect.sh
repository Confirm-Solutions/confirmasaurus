#!/bin/bash

# Connects to the terraform 

URL=$(terraform output -raw ec2instance)
# alternative command: aws ec2 describe-instances | jq -r ".Reservations[0].Instances[0].PublicDnsName"
USER="ec2-user"

echo -e "\n Adding AWS key to ssh-agent. \n"
eval `ssh-agent -s`
ssh-add - <<< "${AWS_KEY_AWS_KEY_PAIR}"

echo -e "\n Connecting to: $USER@$URL \n"
ssh "$USER@$URL" 