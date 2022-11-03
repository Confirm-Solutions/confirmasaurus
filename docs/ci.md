# Continuous Integration

Related: [docs/packaging.md](./packaging.mdci.md)

## Tools
- renovatebot is used to maintain dependency version:
	- dependency dashboard: https://github.com/Confirm-Solutions/confirmasaurus/issues/103
	- app dashboard: https://app.renovatebot.com/dashboard#github/Confirm-Solutions/confirmasaurus
- github actions! See the `.github/workflows` folder.
- pre-commit. See `.pre-commit-config.yaml`.

## Clearing the Github Actions cache

We cache lots of stuff. If you need to clear the cache, run this command. 

```
gh actions-cache list --limit 100 | tail -n +5 | awk '{print $1}' | tr '\n' '\0' | xargs -0 -n1 gh actions-cache delete --confirm
```

