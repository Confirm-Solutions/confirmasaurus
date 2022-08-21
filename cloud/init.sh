#!/bin/bash
sudo yum update -y
sudo yum install -y docker jq
sudo service docker start
sudo usermod -a -G docker ec2-user

# https://stackoverflow.com/questions/51597492/how-to-get-aws-account-number-id-based-on-ec2-instance-which-is-hosted-in-amazo
ACCOUNT=$(aws sts get-caller-identity | jq -r .Account)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker pull $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/${image_name}