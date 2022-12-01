import boto3
import cloudpickle


def f():
    import numpy as np

    A = np.ones(4)
    return A.sum()


s3 = boto3.resource("s3")


def main():
    data = cloudpickle.dumps(f)
    s3.Object("imprint-dump", "jobs/testjob.pkl").put(Body=data)


if __name__ == "__main__":
    main()
