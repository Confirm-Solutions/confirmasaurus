variable ami {
  type        = string
  # default     = "ami-090fa75af13c156b4"
  default     = "ami-040d909ea4e56f8f3" # ecs-optimized ami
  description = "AMI for the instance"
}

variable key-name {
    type        = string
    default     = "aws-key-pair"
    description = "Name of the key pair to use"
}
