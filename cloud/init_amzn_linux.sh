#!/bin/bash

# The update slows down instance creation with little benefit for our
# non-security-focused setup.
# sudo yum update -y

sudo yum install -y docker jq
sudo service docker start
sudo usermod -a -G docker ec2-user

# Everything below here is not needed anymore now that we're using Remote-Containers.
# sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
# sudo chmod +x /usr/local/bin/docker-compose
# # https://stackoverflow.com/questions/51597492/how-to-get-aws-account-number-id-based-on-ec2-instance-which-is-hosted-in-amazo
# export ACCOUNT=$(aws sts get-caller-identity | jq -r .Account)

# # pull the docker image!
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
# docker pull $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/${image_name}

# # give the docker image a nickname so we don't need the big long name.
# docker tag "$ACCOUNT".dkr.ecr.us-east-1.amazonaws.com/${image_name}:latest ${image_name}:latest