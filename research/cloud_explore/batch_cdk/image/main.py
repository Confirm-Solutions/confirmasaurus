import os

import boto3
import cloudpickle

s3 = boto3.resource("s3")


def main():
    bucket = os.environ.get("S3_BUCKET", "imprint-dump")
    filename = os.environ.get("S3_FILENAME", "jobs/testjob.pkl")
    data = s3.Bucket(bucket).Object(filename).get()["Body"].read()
    f = cloudpickle.loads(data)
    print(f)
    return f()


if __name__ == "__main__":
    print(main())
