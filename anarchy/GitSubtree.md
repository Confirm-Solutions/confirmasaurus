# Version control stuff 7/9/22

## Dealing with our open source library is non-trivial.

We're a small team and it'd be nice to just use a single repo for everything. But, we can't because we want to open source `imprint`. Unfortunately, there are no *good* tools for splitting repos apart but still treating them as a single repo. The tool we use to split the repo is `git subtree`. See here for a nice introduction:
[An introduction to git subtree](https://www.atlassian.com/git/tutorials/git-subtree)

I followed the directions under "Adding the sub-project as a remote":

```
git remote add -f imprint git@github.com:Confirm-Solutions/imprint.git
git subtree add --prefix imprint imprint main --squash
```
The `--prefix=imprint` tells git what directory to treat as a subtree. The next `imprint` is the remote we use. 

That code only needs to be run once. So you don't need to worry about this.

Now, for the most part, we can act like the open source external repo does not exist. This is nice. The painful part comes when we want to send our internal changes back to the external repo.
## Contributing subtree changes back upstream

For the foreseeable future, I expect almost all changes to imprint to occur internal to Confirm. So, a subtree fits our use case nicely because we can sort of ignore that the external repo exists and just push new changes to the external open source repo occasionally.

**Currently**, the process looks like:

To update the internal confirmasaurus/imprint subdirectory from changes in the external repo:
```
git remote add -f imprint git@github.com:Confirm-Solutions/imprint.git
git fetch imprint main
git subtree pull --prefix imprint imprint main --squash
```

Add to push a branch to the external repo.
```
git remote add -f imprint git@github.com:Confirm-Solutions/imprint.git
git subtree push --prefix=imprint imprint branchname
```

(I put the remote add commands in there just to avoid errors. You only need to run the remote add once per git clone.)

The process will get a bit more complex in the future if there are external developers who are submitting PRs to imprint. To handle this, *at that point in time*, we should:

1. look around for new tools that solve this nicely.
2. try setting up some CI tools that automatically keep the repos in sync. I think this could be quite easy if we follow some simple branch name conventions.

## Useful references

* https://gist.github.com/SKempin/b7857a6ff6bddb05717cc17a44091202
* https://github.com/joelparkerhenderson/monorepo-vs-polyrepo