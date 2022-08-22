#!/bin/bash

# Connects to the instance launched by terraform 

URL=$(terraform output -raw ec2instance)
# alternative command: aws ec2 describe-instances | jq -r ".Reservations[0].Instances[0].PublicDnsName"
USER="ec2-user" # for use with amazon linux
# USER="ubuntu" # for use with ubuntu

echo -e "\n Connecting to: $USER@$URL \n"
ssh -A "$USER@$URL" 