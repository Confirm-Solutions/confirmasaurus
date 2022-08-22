variable "ami" {
  type        = string
  default     = "ami-090fa75af13c156b4" # default amazon linux ami
  # default     = "ami-052efd3df9dad4825" # ubuntu 22.04 lts
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