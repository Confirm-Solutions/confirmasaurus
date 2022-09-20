#!/bin/bash
# Install/launch docker and set up the ec2-user for using docker.
sudo yum update -y
sudo yum install -y docker jq

# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum install --disablerepo="*" --enablerepo="libnvidia-container" nvidia-container-toolkit -y

sudo service docker start
sudo usermod -a -G docker ec2-user