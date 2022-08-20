# Running jobs on AWS

## Things I've been doing today:

1. Use the `smalldev` docker image!
2. Create a private key. `ssh-add private_key.pem`.
   - Maybe add the key to your codespaces secret keys if you want to directly launch AWS from codespaces.
3. Launch an instance with the ec2 dashboard. (set yourself a reminder to kill the instance at some point.)
4. ssh into the instance (i ran into a problem here where TCP traffic on port 22 was not allowed in my "Security Group")
5. terminate the instance using the ec2 dashboard.
6. Install the awscli: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
7. `aws configure`: go to the IAM dashboard online and find the "Access keys". Create a new access key for your machine. Use it when you run `aws configure`. Now that AWS CLI v2 should be set up on your machine.
8. There is an example here that goes through what is necessary to launch an instance from the CLI: https://docs.aws.amazon.com/vpc/latest/userguide/vpc-subnets-commands-example.html
   - it's quite a lot! So, I decided not to do it and to try using terraform instead.

## Dockerhub

Our repository https://hub.docker.com/repository/docker/tbenthompson/confirm-images

Don't upload sensitive stuff here because it's all public! It's okay for our images to be public because they just have standard boring installation instructions. But make sure you don't commit keys or passwords to docker image and then upload that image publically.
