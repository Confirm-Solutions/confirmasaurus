#!/usr/bin/env python3
import os

import aws_cdk as cdk
import keyring
from aws_cdk import aws_batch_alpha as batch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import Stack
from constructs import Construct


class BatchCdkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        my_vpc = ec2.Vpc.from_lookup(self, "VPC", is_default=True)
        my_sg = ec2.SecurityGroup.from_security_group_id(
            self, "SG", security_group_id="sg-0d8becd8b96dbd11d"
        )

        docker_image_asset = aws_ecr_assets.DockerImageAsset(
            self, "ECRDockerImageAsset", directory="./image"
        )
        image = ecs.ContainerImage.from_docker_image_asset(docker_image_asset)

        batch_job_definition = batch.JobDefinition(  # noqa: F841
            self,
            "JobDefinition",
            container=batch.JobDefinitionContainer(image=image),
        )

        compute_env = batch.ComputeEnvironment(
            self,
            "ComputeEnv",
            compute_resources=batch.ComputeResources(
                minv_cpus=0,
                desiredv_cpus=0,
                maxv_cpus=16,
                vpc=my_vpc,
                security_groups=[my_sg],
                instance_types=[
                    ec2.InstanceType("c6a.large"),  # 2 vCPU, 4 GB RAM
                    ec2.InstanceType("c6a.xlarge"),  # 4 vCPU, 8 GB RAM
                    ec2.InstanceType("c6a.8xlarge"),  # 32 vCPU, 64 GB RAM
                    ec2.InstanceType("m6a.large"),  # 2 vCPU, 8 GB RAM
                    ec2.InstanceType("m6a.8xlarge"),  # 32 vCPU, 128 GB RAM
                    ec2.InstanceType("p3.2xlarge"),  # 8 vCPU, 61 GB RAM, 1 V100
                    ec2.InstanceType("p3.8xlarge"),  # 32 vCPU 244 GB RAM, 4 V100
                ],
                type=batch.ComputeResourceType.SPOT,
            ),
        )
        queue = batch.JobQueue(  # noqa: F841
            self,
            "JobQueue",
            compute_environments=[
                batch.JobQueueComputeEnvironment(
                    compute_environment=compute_env, order=1
                )
            ],
            priority=1,
        )


aws_account_id = keyring.get_password("aws-confirm-account-id", os.environ["USER"])
env = cdk.Environment(account=aws_account_id, region="us-east-1")
app = cdk.App()
BatchCdkStack(app, "batch-cdk", env=env)

app.synth()

# my_instance = ec2.Instance(
#     self,
#     "Instance",
#     instance_type=ec2.InstanceType("t3.micro"),
#     machine_image=ec2.MachineImage.latest_amazon_linux(),
#     vpc=my_vpc,
#     security_group=my_sg,
# )
