# Tests and Continuous Integration

Related: [docs/packaging.md](./packaging.md)

## Slow tests

Use `@pytest.mark.slow` to indicate that a test is slow and should not be run by default. To run slow tests, add the `--run-slow` flag to pytest. Slow tests will not be run by the fast CI on PRs but will be run by the nightly end to end tests.

## Nightly end to end tests

Our end to end tests are run nightly via the [.github/workflows/e2e.yml](../.github/workflows/e2e.yml) actions workflow which launches [confirm/cloud/modal_test_runner.py](../confirm/cloud/modal_test_runner.py). That script runs all tests including slow tests on a Modal server with a single GPU.

## Tools
- renovatebot is used to maintain dependency version:
	- dependency dashboard: https://github.com/Confirm-Solutions/confirmasaurus/issues/103
	- app dashboard: https://app.renovatebot.com/dashboard#github/Confirm-Solutions/confirmasaurus
- github actions! See the `.github/workflows` folder.
- pre-commit. See `.pre-commit-config.yaml`.

## Clearing the Github Actions cache

We cache lots of stuff including our Python environments. If you need to clear the cache, run this command. 

```
gh actions-cache list --limit 100 | tail -n +5 | awk '{print $1}' | tr '\n' '\0' | xargs -0 -n1 gh actions-cache delete --confirm
```

