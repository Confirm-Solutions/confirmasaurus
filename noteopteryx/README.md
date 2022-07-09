# README

[![hackmd-github-sync-badge](https://hackmd.io/HslnB140Sai5C-HMS6uJIQ/badge)](https://hackmd.io/HslnB140Sai5C-HMS6uJIQ)

This is a repo for internal technical notes and updates. NOT documentation. Documentation is kept up to date and should live elsewhere. These notes might be updated, but I'm going to treat this repo like the wild west. Just push anything anytime, no reviews. Just a place to share. 

I'll probably mostly push markdown here. Markdown is nice because there are many nice editors, including collaborative ones (like HackMD, Gitbook) and the GitHub interface renders Markdown very nicely. Also, the Slack interface renders GitHub Markdown links fairly well, so it's a nice format to give previews for sharing technical docs on slack.

## Why not google docs or overleaf or ...?

Because I am fallible. Using google docs is great! See [here](https://drive.google.com/drive/u/0/folders/1GMlPXNFFXWg-NvYmo8UvNZfJhU54kObx) for the company google drive folder which has lots of good stuff. Basically:

* google docs sucks for math.
* google docs has much higher activation energy compared to dropping some ideas in a markdown doc in my already-open editor.

Overleaf is also great, but it's pretty heavyweight for quick note-taking that I do all the time. 

## Why is this kept as a separate repo?

It's not really. It's going to be used through the main `research` repo via `git subtree`. Using `git subtree` to maintain a monorepo is an experiment that I (Ben) would like to try. But for these notes, it's nice to _also_ keep them as a separate repo so that the whole repo can be loaded up via git/github sync into various collaborative markdown editors or local editor apps.

## How to use this repo?

* Just pull, commit, push. Don't worry about reviews. These are intended to be working notes.
* Use whatever markdown editor that you like:
  * The GitHub online interface is actually totally fine. Just click on the little edit pencil icon!
  * I've been using [Obsidian](https://obsidian.md). It's quite nice and I use it a lot for my personal life. There's a git auto-sync feature available. The merge conflict handling is not bad at all!
  * Any typical code editor/IDE is fine! I like VSCode.
  * [HackMD](hackmd.io) is really cool: collaborative editing/commenting. I added a nice badge to the top of this doc that let's you open the file directly into hackmd.
  * I do not recommend GitBook.com even though it is also a really cool option since it allows for collaborative editing and commenting kind of like google docs. But, the github sync is really fucked up if there are any merge conflicts - it completely deleted some files and renamed some others in nonsensical ways!

## How to share GitHub markdown docs in slack?

By default, if you drop a GitHub link to a public github repo, there will be a preview.
But, this markdown link preview doesn't display for links to markdown files in private repos. This can be fixed by inviting the GitHub user to the channel you're in using `\invite @github`. You might also need to authorize the GitHub user for the relevant repo, but I've already done this for the Confirm-Solutions organization.

## Math test

Yeah, this works!

$$x = \exp(y)$$

$$
f(x) = x * e^{2 pi i \xi x}
$$
