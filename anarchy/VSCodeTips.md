- Load up VSCode from the root directory. 
- Install the Python and jupyter extension
- Set up unit tests:
	- useful shortcuts: 
		- "cmd + ;" is the prefix for most testing commands
			- "cmd + ;" then "c" runs the test under the cursor
			- "cmd + ;" then "l" runs the last test(s) you ran
			- "cmd + ;" then "f" runs all the tests in the file
			- "cmd + ;" then "e" runs all failed tests
	- you can also run tests in debug mode and set breakpoints, very helpful
	- Open the "Output" panel and go to "Python test log" to see the raw pytest output.
	- If you get "pytest discovery error", go to "Python" in "Output" and find the error and fix it.
![[Pasted image 20220710170910.png]]
- the neovim and vim plugins are great if you're into that.
- https://code.visualstudio.com/docs/languages/python
- https://realpython.com/advanced-visual-studio-code-python/
- getting nice docstrings, install the auto docstring extension and set: `"autoDocstring.docstringFormat": "google-notypes"`
	- why use google docstring style? they are just prettier. the numpy docstring style is also nice, but it's a bit more verbose and takes up a lot of space. https://sphinxcontrib-napoleon.readthedocs.io/en/latest/

## Using R with VSCode

- The Codespaces Dockerfile has everything set up already! But here's what I did:
	- I installed R and packages via the cran debian apt ppa. On Mac, I would just install R via homebrew and then using `install.packages()` because that will install binary packages. However, on linux, `install.packages()` doesn't use binary packages.
	- The R extension is essential. I followed the official instructions.
	- I also installed radian and httpgd. https://code.visualstudio.com/docs/languages/r
- Rmd files seem to work well, but Jupyter notebooks can be nice because they put plots inline. 
- if R is installed and the `IRKernel` package is installed, you should be able to install the Jupyter kernel with `R -e "IRkernel::installspec()"`. This is already set up in the Dockerfile.
- The "Jupyter" output panel in VSCode contains stuff produced by subprocesses. R-INLA runs a command line utility that does most of the computation so you need to look here for output. There might be a way to get this output into the notebook interface, but I don't know how.