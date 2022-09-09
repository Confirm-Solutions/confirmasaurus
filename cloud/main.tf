terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "us-east-1"
}

locals {
  vars = {
    image_name = var.image_name
  }
}

resource "aws_iam_role" "main" {
  name               = "main-${terraform.workspace}"
  assume_role_policy = file("assumerolepolicy.json")
}

resource "aws_iam_policy" "main" {
  name        = "main-${terraform.workspace}"
  description = "IAM policy for individual devenvironments."
  policy      = file("iampolicy.json")
}

resource "aws_iam_policy_attachment" "main" {
  name       = "main-${terraform.workspace}"
  roles      = ["${aws_iam_role.main.name}"]
  policy_arn = aws_iam_policy.main.arn
}

resource "aws_iam_instance_profile" "main" {
  name = "main-${terraform.workspace}"
  role = aws_iam_role.main.name
}


resource "aws_instance" "main" {
  ami                  = var.ami
  instance_type        = var.instance_type
  key_name             = var.key_name
  iam_instance_profile = aws_iam_instance_profile.main.name

  root_block_device {
    volume_size = 40
    volume_type = "gp3"
  }

  tags = {
    Name = "confirmasaurus"
  }
  
# Here, we are using user_data to run an install script that will be the first
# command run on the new instance.
#   these two links explain how to use user data to install things on the
#   instance:
# - https://klotzandrew.com/blog/deploy-an-ec2-to-run-docker-with-terraform
# - https://awstip.com/to-set-up-docker-container-inside-ec2-instance-with-terraform-3af5d53e54ba
  user_data = templatefile("init_amzn_linux.sh", local.vars)
}

output "ec2instance" {
  value = aws_instance.main.public_dns
}

output "id" {
  value = aws_instance.main.id 
}