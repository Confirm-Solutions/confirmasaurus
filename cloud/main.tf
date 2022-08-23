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

resource "aws_iam_role" "ec2_role" {
  name               = "ec2_role"
  assume_role_policy = file("assumerolepolicy.json")
}

resource "aws_iam_policy" "policy" {
  name        = "test-policy"
  description = "A test policy"
  policy      = file("iampolicy.json")
}

resource "aws_iam_policy_attachment" "test-attach" {
  name       = "test-attachment"
  roles      = ["${aws_iam_role.ec2_role.name}"]
  policy_arn = aws_iam_policy.policy.arn
}

resource "aws_iam_instance_profile" "test_profile" {
  name = "test_profile"
  role = aws_iam_role.ec2_role.name
}


resource "aws_instance" "app" {
  ami                  = var.ami
  instance_type        = var.instance_type
  key_name             = var.key_name
  iam_instance_profile = aws_iam_instance_profile.test_profile.name

  root_block_device {
    volume_size = 40
    volume_type = "gp3"
  }

  tags = {
    Name = "confirmasaurus"
  }
  
#   these two links explain how to use user data to install things on the instance:
# - https://klotzandrew.com/blog/deploy-an-ec2-to-run-docker-with-terraform
# - https://awstip.com/to-set-up-docker-container-inside-ec2-instance-with-terraform-3af5d53e54ba
  user_data = templatefile("init_amzn_linux.sh", local.vars)
}

output "ec2instance" {
  value = aws_instance.app.public_dns
}

output "id" {
  value = aws_instance.app.id 
}