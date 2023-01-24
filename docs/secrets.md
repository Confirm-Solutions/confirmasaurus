# Secrets

We store our secrets in [this encrypted file in the git repo](../test_secrets.enc). This includes passwords and tokens for various cloud services.

[How is this safe? This post argues why this is actually one of the most safe things to do with secrets.](https://oteemo.com/hashicorp-vault-is-overhyped-and-mozilla-sops-with-kms-and-git-is-massively-underrated/)

The tools involved are:
- [AWS Key Management Service (KMS)](https://aws.amazon.com/kms/) is used to generate the encryption key for the secrets file. We generate this key with a brief CDK script [here](../cloud/sops_kms).
- [Mozilla SOPS](https://github.com/mozilla/sops) configured with [`.sops.yaml`](../.sops.yaml).

## Installing SOPS

```
brew install sops
```

## Decrypting the secrets

Log into AWS using `aws sso configure`. Please do not stray far from these commands. 

Then decrypt the secrets with:
```
sops -d test_secrets.enc.env > .env
```

## Encrypting new secrets

Log into AWS using `aws sso configure`. Please do not stray far from these commands. 

First, decrypt the secrets as above. Then, modify the secret or add a secret or whatever. Then, encrypt the new secrets with:

```
sops -e .env > test_secrets.enc.env
```

## Accessing secrets in code

Use the `getenv` package to load the `.env` file and access its entries. For example:

```
import getenv
host = getenv.dotenv_values()['CLICKHOUSE_HOST']
```

## Decrypting secrets in GitHub actions

I followed [the directions here](https://www.automat-it.com/post/using-github-actions-with-aws-iam-roles) to set up GitHub Actions to access KMS and decrypt the secrets. I created a permissions policy in AWS named "AccessSecrets" following [the instructions here](https://github.com/mozilla/sops#assuming-roles-and-using-kms-in-various-aws-accounts). I created a "GitHubActionsRole" that has that permissions policy attached.

## Decrypting secrets in other services (e.g. Modal)

The "AccessSecrets" IAM User can be used for decrypting secrets.