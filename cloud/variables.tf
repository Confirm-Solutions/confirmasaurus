variable "ami" {
  type        = string
  default     = "ami-090fa75af13c156b4" # default amazon linux ami
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
  default     = "t2.xlarge"
  # default     = "p3.2xlarge"
  description = "description"
}