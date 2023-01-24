import shutil

import aws_cdk as cdk
from aws_cdk import aws_batch_alpha as batch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import Stack
from constructs import Construct

shutil.copy2("../../../pyproject.toml", "./image")
shutil.copy2("../../../poetry.lock", "./image")


class InstanceProfile(Construct):
    """
    Custom construct for the Instance Profile resource.
    Used to wrap the Instance Role construct.
    """

    @property
    def profile_arn(self):
        if self._instance is None:
            self._instance = self._create_instance()
        return self._instance.attr_arn

    def attach_role(self, role):
        self._roles.append(role.role_name)

    def _create_instance(self):
        return iam.CfnInstanceProfile(
            self, self._id + "cfn-instance-profile", roles=self._roles
        )

    def __init__(self, scope: Construct, id: str):
        super().__init__(scope, id)
        self._roles = []
        self._instance = None
        self._id = id


class BatchCdkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        my_vpc = ec2.Vpc.from_lookup(self, "VPC", is_default=True)
        my_sg = ec2.SecurityGroup.from_security_group_id(
            self, "SG", security_group_id="sg-0d8becd8b96dbd11d"
        )

        s3_bucket = s3.Bucket(  # noqa: F841
            self,
            "BatchCloudpickles",
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        docker_image_asset = aws_ecr_assets.DockerImageAsset(
            self,
            "ECRDockerImageAsset",
            directory="./image",
            platform=aws_ecr_assets.Platform.LINUX_AMD64,
        )
        image = ecs.ContainerImage.from_docker_image_asset(docker_image_asset)

        batch_job_definition = batch.JobDefinition(  # noqa: F841
            self,
            "JobDefinition",
            container=batch.JobDefinitionContainer(
                image=image,
                vcpus=2,
                memory_limit_mib=1024,
                environment={
                    "S3_BUCKET": s3_bucket.bucket_name,
                },
            ),
        )

        batch_instance_role = iam.Role(
            self,
            "BatchJobInstanceRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"),
                iam.ServicePrincipal("ecs.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
            ],
        )

        batch_instance_profile = InstanceProfile(self, "BatchJobInstanceProfile")
        batch_instance_profile.attach_role(batch_instance_role)

        compute_env = batch.ComputeEnvironment(
            self,
            "ComputeEnv",
            compute_resources=batch.ComputeResources(
                minv_cpus=0,
                desiredv_cpus=0,
                maxv_cpus=16,
                vpc=my_vpc,
                security_groups=[my_sg],
                instance_role=batch_instance_profile.profile_arn,
                instance_types=[
                    ec2.InstanceType("c6i.large"),
                    ec2.InstanceType("c6i.xlarge"),
                    ec2.InstanceType("c6i.2xlarge"),
                    ec2.InstanceType("c6i.4xlarge"),
                    ec2.InstanceType("m6i.large"),
                    ec2.InstanceType("m6i.xlarge"),
                    ec2.InstanceType("m6i.2xlarge"),
                    ec2.InstanceType("m6i.4xlarge"),
                    ec2.InstanceType("r6i.large"),
                    ec2.InstanceType("r6i.xlarge"),
                    ec2.InstanceType("r6i.2xlarge"),
                    ec2.InstanceType("r6i.4xlarge"),
                    ec2.InstanceType("p3.2xlarge"),
                    ec2.InstanceType("p3.8xlarge"),
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


env = cdk.Environment(account="644171722153", region="us-east-1")
app = cdk.App()
BatchCdkStack(app, "batch-cdk", env=env)

app.synth()
