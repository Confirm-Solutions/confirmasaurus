import boto3
import cloudpickle
import numpy as np


def f():
    np.random.seed(0)
    A = np.random.rand(20, 20)
    Ai = np.linalg.inv(A)
    return Ai.diagonal()[0]


s3 = boto3.resource("s3")


def main():
    data = cloudpickle.dumps(f)
    s3.Object("imprint-dump", "jobs/testjob.pkl").put(Body=data)


if __name__ == "__main__":
    print(f())
    main()
