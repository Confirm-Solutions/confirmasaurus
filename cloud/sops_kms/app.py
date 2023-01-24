#!/usr/bin/env python3
import os

import aws_cdk as cdk

from aws_cdk import Stack
import aws_cdk.aws_kms as kms
import aws_cdk.aws_iam as iam
from constructs import Construct

def get_kms_policy_documents():
    policy_document = iam.PolicyDocument()
    policy_statement = iam.PolicyStatement()
    policy_statement.effect.ALLOW
    policy_statement.add_actions('kms:*')
    policy_statement.add_all_resources()
    policy_statement.add_account_root_principal()
    policy_document.add_statements(policy_statement)
    return policy_document

class BatchCdkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        kms_policy_document = get_kms_policy_documents()
        
        _ = kms.Key(self, 
            id='SOPS_key',
            description='Key for SOPS',
            enabled=True,
            enable_key_rotation=False,
            policy=kms_policy_document,
        )


env = cdk.Environment(account='644171722153', region="us-east-1")
app = cdk.App()
BatchCdkStack(app, "sops-key", env=env)

app.synth()

