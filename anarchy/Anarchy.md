# Anarchy

[![hackmd-github-sync-badge](https://hackmd.io/HslnB140Sai5C-HMS6uJIQ/badge)](https://hackmd.io/HslnB140Sai5C-HMS6uJIQ)

## Anarchy: a space for doing ANYTHING

Pull requests are great! But sometimes you just want to do stuff. I see two main uses here:

- notebooks and scratch/prototype code that you don't expect to maintain.
- working notes, like the ones you are currently reading. These are not documentation! I will probably never update this document and that's fine! We can feel free to copy bits from the wild west into more stable, maintained parts of our world.

To repeat, do *not* put permanent documentation here. Documentation is kept up to date and should live elsewhere.

I'll probably mostly push markdown here. Markdown is nice because there are many nice editors, including collaborative ones (like HackMD, Gitbook) and the GitHub interface renders Markdown very nicely. Also, the Slack interface renders GitHub Markdown links fairly well, so it's a nice format to give previews for sharing technical docs on slack.

## Hiding anarchy/ files in github pull requests

[I set up `.gitattributes` so that it'll be easy to ignore files in the anarchy folder when we make PRs.](https://stackoverflow.com/questions/20120478/ignoring-specific-files-file-types-or-folders-in-a-pull-request-diff/54235094)

## Why not google docs or overleaf or ...?

Because I am fallible. Using google docs is great! See [here](https://drive.google.com/drive/u/0/folders/1GMlPXNFFXWg-NvYmo8UvNZfJhU54kObx) for the company google drive folder which has lots of good stuff. Basically:
- google docs suck for math and code.
- google docs has much higher activation energy compared to dropping some ideas in a markdown doc in my already-open editor.
- living inside our git repo is really pleasant.

On the other hand, the collaborative editing and commenting on google docs is awesome. So, pick your poison.
Overleaf is also great, but it's pretty heavyweight for quick note-taking that I do all the time.

## How to share GitHub markdown docs in slack?

By default, if you drop a GitHub link to a public github repo, there will be a preview. I like this so that people can read the first few lines without opening something else. But, this markdown link preview doesn't display for links to markdown files in private repos. This can be fixed by inviting the GitHub user to the channel you're in using `\invite @github`. You might also need to authorize the GitHub user for the relevant repo, but I've already done this for the Confirm-Solutions organization.

## Markdown editor suggestions

- The GitHub online interface is actually totally fine. Just click on the little edit pencil icon!
- I've been using [Obsidian](https://obsidian.md). It's quite nice and I use it a lot for my personal life. There's a git auto-sync feature available. The merge conflict handling is not bad at all!
- Any typical code editor/IDE is fine! I like VSCode.
- [HackMD](hackmd.io) is really cool: collaborative editing/commenting. I added a nice badge to the top of this doc that let's you open the file directly into hackmd.
- I do not recommend GitBook.com even though it is also a really cool option since it allows for collaborative editing and commenting kind of like google docs. But, the github sync is really fucked up if there are any merge conflicts - it completely deleted some files and renamed some others in nonsensical ways!

## Math test

Yeah, this works!

$$x = \exp(y)$$

$$
f(x) = x * e^{2 pi i \xi x}
$$
