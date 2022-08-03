# JAX development patterns

JAX development patterns that might be useful:
- Pull your `jax.jit` and `jax.vmap` calls into the outermost layer of the code. This has two benefits
  - it's easier to write code that is less vectorized.
  - grouping more operations before jit-ing gives more room for the compiler to optimize.
  - `jax.jit(jax.vmap(...` is better than `jax.vmap(jax.jit(...`
- Put a bunch of shared variables into a class and then include `self` in the list of `static_argnums`. This is a useful strategy for having a large number of static args.
- `jax.jit(f).lower(*args).compile()` is a useful snippet for compiling a function without running the function. [There is a long running JAX project to build better ahead-of-time compilation tools.](https://github.com/google/jax/issues/7733)

Techniques for avoiding non-deterministic behavior that will make jax complain:
- `jnp.where(...`
- `jax.lax.fori_loop`, `jax.lax.while_loop`. `jax.lax.cond`: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#structured-control-flow-primitives

Other things:
- understanding pytrees in JAX is really helpful: https://jax.readthedocs.io/en/latest/pytrees.html
- other difficult stuff: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

Scary things:
- sometimes `jax.lax.cond` combined with `jax.vmap` will result in both branches of your `cond` executing. In extreme cases, this can result in infinite loops. See [the problem James and I ran into here](https://github.com/pyro-ppl/numpyro/issues/1461).
