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


resource "aws_instance" "app" {
  ami           = var.ami
  instance_type = "t2.xlarge"
  key_name      = var.key_name

  tags = {
    Name = "confirmasaurus"
  }
  user_data = templatefile("init.sh", local.vars)
}