# Coding and software standards and principles

I wrote this as a little placeholder because I think it'd be good to have some standards for our "product". Ideas for changes or additions or deletions are super duper welcome. 

**We're a small company and we don't want a lot of rules.**

- We should probably just have one repo for everything at the moment. This is not a commitment to be a monorepo company, just an acknowledgement that we're small and more than one repo will incur coordinations costs.
- Important code should probably be reviewed.
- You should probably be pair programming more than you are.
- Iteration time is unimaginably important.
- Pretty much anything goes in the `research` folder.
- Tests are critical to correctness, but try not to let testing get in the way of doing stuff.
- Use `black` for formatting.
- Follow PEP8 suggestions where possible. This is almost entirely automated by `flake8`.
- Type annotations are great, especially for major interface boundaries.
- Write some documentation for major interface boundaries, mainly as docstrings and end-to-end usage tutorials.
- Use ["google-notypes"](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e) as our docstring format.
