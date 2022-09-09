variable "ami" {
  type        = string
  default     = "ami-0853175484e1b2c5e" # amazon linux 2 with nvidia drivers
  description = "AMI for the instance"
}

variable "key_name" {
  type        = string
  default     = "aws-key-pair"
  description = "Name of the key pair to use"
}

variable "image_name" {
  type        = string
  default     = "smalldev:latest"
  description = "Name of the docker image on ECR to use."
}

variable "instance_type" {
  type        = string
  # default     = "t2.xlarge"
  default     = "p3.2xlarge"
  description = "description"
}