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
  assume_role_policy = "${file("assumerolepolicy.json")}"
}

resource "aws_iam_policy" "policy" {
  name        = "test-policy"
  description = "A test policy"
  policy      = "${file("iampolicy.json")}"
}

resource "aws_iam_policy_attachment" "test-attach" {
  name       = "test-attachment"
  roles      = ["${aws_iam_role.ec2_role.name}"]
  policy_arn = "${aws_iam_policy.policy.arn}"
}

resource "aws_iam_instance_profile" "test_profile" {
  name  = "test_profile"
  role = "${aws_iam_role.ec2_role.name}"
}


resource "aws_instance" "app" {
  ami           = var.ami
  instance_type = "t2.xlarge"
  key_name      = var.key_name
  iam_instance_profile = "${aws_iam_instance_profile.test_profile.name}"
  
  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    encrypted   = true
    kms_key_id  = data.aws_kms_key.customer_master_key.arn
  }

  tags = {
    Name = "confirmasaurus"
  }
  user_data = templatefile("init.sh", local.vars)
}

output "ec2instance" {
  value = aws_instance.app.public_dns
}