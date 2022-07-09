# Confirmasaurus


## Mono-repo?

Because it's less effort!

We still split out some pieces of the repo into separate components because they have special needs:
- The `imprint` library needs to be open-source!
- The `noteopteryx` repo is nice to be able to load up separately. (This might change.)

The tool we use to split the repo is `git subtree`. See here for a nice introduction:
[An introduction to git subtree](https://www.atlassian.com/git/tutorials/git-subtree)

I followed the directions under "Adding the sub-project as a remote".
