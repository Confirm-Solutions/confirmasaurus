# Confirmasaurus


## Mono-repo?

Because it's less effort! We still split out `imprint` since it needs to be open source!

The tool we use to split the repo is `git subtree`. See here for a nice introduction:
[An introduction to git subtree](https://www.atlassian.com/git/tutorials/git-subtree)

I followed the directions under "Adding the sub-project as a remote":

```
git remote add -f imprint git@github.com:Confirm-Solutions/imprint.git
git subtree add --prefix imprint imprint main --squash
```

To update the repo:
```
git fetch imprint main
git subtree pull --prefix imprint imprint main --squash
```

In the "Contributing back upstream" section, you should ignore the comments about forking, because we're all maintainers/collaborators on the imprint repo so we don't need to fork to do a subtree push:
```
git subtree push --prefix=imprint imprint main
```

Common situations that I'd like to accomodate:
- 