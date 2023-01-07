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

s3 = boto3.resource("s3")
client = boto3.client("batch")


def serialize_to_s3(
    f, bucket="batch-cdk-batchcloudpickles4708249b-1y12z8ci4k38p", filename=None
):
    data = cloudpickle.dumps(f)
    if filename is None:
        filename = f"{str(uuid.uuid4())}.pkl"
    s3.Object(bucket, filename).put(Body=data)
    return bucket, filename


def submit_job(bucket, filename, cpus, memory, gpu):
    resourceRequirements = [
        {"type": "VCPU", "value": str(cpus)},
        {"type": "MEMORY", "value": str(memory)},
    ]
    if gpu:
        resourceRequirements.append({"type": "GPU", "value": "1"})
    return client.submit_job(
        # TODO: get definition, queue names from cdk.
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
    bucket, filename = serialize_to_s3(f)
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
    bucket, filename = serialize_to_s3(f)
    response = submit_job(bucket, filename, cpus, memory, gpu)
    return response, bucket, filename


def include_package(package):
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


if __name__ == "__main__":
    submit_job("imprint-dump", "jobs/testjob.pkl")
