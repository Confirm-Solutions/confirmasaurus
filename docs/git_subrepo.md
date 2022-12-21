# Version control stuff

Originally written 7/9/22 based on git subtree, completely revamped 12/19/22 based on git subrepo. Much cleaner!

## Dealing with our open source library is non-trivial.

We're a small team and it'd be nice to just use a single repo for everything. But, we can't because we want to open source `imprint`. Unfortunately, there are no *good* tools for splitting repos apart but still treating them as a single repo. The options are:
1. `git submodule`
2. `git subtree`
3. [`git subrepo`](https://github.com/ingydotnet/git-subrepo)

I like [the discussion here by the `git subrepo` author on why `subrepo` is the best option](https://github.com/ingydotnet/git-subrepo/blob/master/Intro.pod).

As far as I can tell, the design of `git subrepo` is pretty close to the optimal design for our situation. The only downside to the tool is that it's not perfectly maintained. I think that's going to be okay for us. 

For the most part, while developing confirm we can act like the open source imprint repo does not exist. This is nice. 

## Automated pushes from confirm to imprint

See [the actions workflow here](../.github/workflows/test.yml). It pushes
changes to the imprint branch `sync` and then automatically creates a PR to
merge those changes.

The flow from confirm to imprint looks like:
1. (manual) Merge PR to main in confirm
2. (automatic) subrepo push to imprint
3. (automatic) PR created on imprint
4. (automatic) automatically push the .gitrepo file to confirm
5. (manual) improve/merge the PR on imprint

I'd like to have a similar reverse flow from imprint to confirm but it's not
implemented yet. It would probably need to be implemented as an actions
workflow on the imprint repo triggering a workflow in the confirmasaurus repo.
The end of the discussion here would be useful for building this. 
https://github.com/orgs/community/discussions/26323

## Contributing changes to imprint

```
git subrepo push imprint --branch sync --squash --debug --verbose --message "Subrepo push to imprint"
```

## Pulling changes from imprint

```
git subrepo pull imprint --debug --verbose --message "Pull from imprint to confirm"
```

## (historical) What I did to set up subrepo

`git subrepo` has rough edges and some bugs. This is acceptable because the
`git subtree` and `git submodule` alternatives are so shitty. One of the
greatest gifts an engineer could give the world would be to fix this basic
infrastructure!

1. `git subrepo init imprint`
2. Manually open the `imprint/.gitrepo` file and set:
    - `remote = git@github.com:Confirm-Solutions/imprint.git`
	- `branch = main`
	- `commit = go-get-the-commit-id-from-imprint-repo`
	- `parent = go-get-the-commit-id-from-confirm-repo`
3. Manually implement this fix: https://github.com/ingydotnet/git-subrepo/pull/498/files

## Useful references

- https://github.com/ingydotnet/git-subrepo/wiki/Basics
- https://gist.github.com/SKempin/b7857a6ff6bddb05717cc17a44091202
- https://github.com/joelparkerhenderson/monorepo-vs-polyrepos
- if you ever screw up... https://stackoverflow.com/questions/63496368/git-how-to-remove-all-files-from-the-git-history-that-are-not-currently-prese