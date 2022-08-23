#!/bin/bash
# Install/launch docker and set up the ec2-user for using docker.
sudo yum update -y
sudo yum install -y docker jq
sudo service docker start
sudo usermod -a -G docker ec2-user