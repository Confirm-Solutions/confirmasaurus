import io
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path

import boto3
import cloudpickle


def _serialize_to_s3(
    f, bucket="batch-cdk-batchcloudpickles4708249b-1y12z8ci4k38p", filename=None
):
    s3 = boto3.resource("s3")
    data = cloudpickle.dumps(f)
    if filename is None:
        filename = f"{str(uuid.uuid4())}.pkl"
    s3.Object(bucket, filename).put(Body=data)
    return bucket, filename


def _submit_job(bucket, filename, cpus, memory, gpu):
    resourceRequirements = [
        {"type": "VCPU", "value": str(cpus)},
        {"type": "MEMORY", "value": str(memory)},
    ]
    if gpu:
        resourceRequirements.append({"type": "GPU", "value": "1"})
    client = boto3.client("batch")
    return client.submit_job(
        # TODO: get definition, queue names from cdk?
        jobDefinition="JobDefinition24FFE3ED-aa3a6b0e573391b",
        jobName="job1",
        jobQueue="JobQueueEE3AD499-0FVaqWBM4g2kOnBf",
        containerOverrides={
            "resourceRequirements": resourceRequirements,
            "environment": [
                {
                    "name": "S3_BUCKET",
                    "value": bucket,
                },
                {"name": "S3_FILENAME", "value": filename},
            ],
        },
    )


def local_test(f, *args, **kwargs):
    """
    This is a mock of remote_run that runs that function locally instead of on
    AWS Batch.

    Args:
        f: The function/closure to run.
    """
    bucket, filename = _serialize_to_s3(f)
    home_dir = os.environ["HOME"]
    subprocess.call(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            f"S3_BUCKET={bucket}",
            "-e",
            f"S3_FILENAME={filename}",
            "-v",
            f"{home_dir}/.aws:/root/.aws",
            "jobrunner",
        ]
    )


def remote_run(f, *, cpus, memory, gpu):
    """
    Run a function/closure on AWS Batch. The function will be serialized to a
    file in an S3 bucket and then downloaded and run on the AWS Batch instance.

    For more details on the acceptable values for cpus, memory, and gpu, see
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.submit_job

    Args:
        f: _description_
        cpus: The number of CPUs to request.
        memory: The amount of memory to request in MiB.
        gpu: Should we request a GPU?

    Returns:
        The response from the AWS Batch API, the bucket name, and the filename.
    """
    bucket, filename = _serialize_to_s3(f)
    response = _submit_job(bucket, filename, cpus, memory, gpu)
    return response, bucket, filename


def include_package(package):
    """
    A decorator that will include a package in a zip file that is wrapped up
    into the closure that is passed to remote_run.

    Args:
        package: The package to include. This should be the package object not
        a string.

    Returns:
        The wrapped function.
    """
    package_name = package.__name__
    parent_dir = Path(package.__file__).parent
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_basename = os.path.join(tmpdir, package_name)
        shutil.make_archive(zip_basename, "zip", parent_dir)
        with open(f"{zip_basename}.zip", "rb") as f:
            bytes = f.read()

    def wrapper(f):
        def wrapped():
            zip = zipfile.ZipFile(io.BytesIO(bytes))
            zip.extractall(f"./{package_name}")
            return f()

        return wrapped

    return wrapper
