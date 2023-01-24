import os
import modal
import dotenv

# deploy with `modal app deploy cloud/sops_kms/set_modal_secrets.py::prod_stub`
prod_secrets = dotenv.dotenv_values("cloud/sops_kms/prod_secrets.gitignore.env")
prod_stub = modal.Stub("confirm-prod-secrets")
prod_stub["secret"] = modal.Secret(prod_secrets)

# deploy with `modal app deploy cloud/sops_kms/set_modal_secrets.py::test_stub`
test_secrets = dotenv.dotenv_values("test_secrets.gitignore.env")
test_stub = modal.Stub("confirm-test-secrets")
test_stub["secret"] = modal.Secret(test_secrets)