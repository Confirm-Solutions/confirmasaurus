import os

import keyring
import modal


secrets = dict()
for (keyring_name, env_name) in [
    ("clickhouse-confirm-test-host", "CLICKHOUSE_HOST"),
    ("clickhouse-confirm-test-host", "CLICKHOUSE_TEST_HOST"),
    ("clickhouse-confirm-test-password", "CLICKHOUSE_PASSWORD"),
    ("upstash-confirm-coord-test-host", "REDIS_HOST"),
    ("upstash-confirm-coord-test-password", "REDIS_PASSWORD"),
]:
    secrets[env_name] = keyring.get_password(keyring_name, os.environ["USER"])

stub = modal.Stub("confirm-secrets")
stub["secret"] = modal.Secret(secrets)
