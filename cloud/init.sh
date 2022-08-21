#!/bin/bash
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

ACCOUNT=$(aws sts get-caller-identity | jq -r .UserId)
docker pull $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/${image_name}