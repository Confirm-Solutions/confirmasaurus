# AWS Batch Job Runner.

What are we here to do?
- Launching an AWS Batch cluster that will run approximately arbitrary functions.
- Why not use AWS Lambda? Lambda doesn't run on GPUs!

Useful documentation:

- a fully worked example for using AWS Batch. This code is mostly based off this example:
  https://github.com/aws-samples/aws-cdk-deep-learning-image-vector-embeddings-at-scale-using-aws-batch/blob/main/batch_job_cdk/stack.py
- boto3 docs for AWS Batch:
  https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html
- AWS CDK docs for AWS Batch:
  https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_batch_alpha.html


##Setting up AWS CDK:

https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html

Installed nvm: node version manager

```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
```

Restart terminal to use nvm and then install Node 18.

```
nvm install 18
```

Upgrade npm:

```
npm install -g npm
```

Install AWS CDK:

```
npm install -g aws-cdk
```

Follow the "Bootstrapping" instructions in the docs here: 
https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html

## Deployment

- launch docker desktop. docker desktop on mac sucks but don't cry too much.
- `docker pull ghcr.io/confirm-solutions/smalldev:latest`
- `docker tag ghcr.io/confirm-solutions/smalldev:latest 644171722153.dkr.ecr.us-east-1.amazonaws.com/smalldev:latest`
- `cdk deploy`