import os
import time

import boto3
import cloudpickle

s3 = boto3.resource("s3")


def main():
    print("Loading job from S3")
    start = time.time()
    bucket = os.environ.get("S3_BUCKET", "imprint-dump")
    filename = os.environ.get("S3_FILENAME", "jobs/testjob.pkl")
    data = s3.Bucket(bucket).Object(filename).get()["Body"].read()
    f = cloudpickle.loads(data)
    print("Loaded job from S3 in {:.2f} seconds".format(time.time() - start))
    print("Running job")
    start = time.time()
    out = f()
    print("Ran job in {:.2f} seconds".format(time.time() - start))
    return out


if __name__ == "__main__":
    print(main())
