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
- useful issues talking about using JAX with async/await:
  - https://github.com/google/jax/issues/6772
  - https://github.com/google/jax/issues/3769

Scary things:

- sometimes `jax.lax.cond` combined with `jax.vmap` will result in both branches of your `cond` executing. In extreme cases, this can result in infinite loops. See [the problem James and I ran into here](https://github.com/pyro-ppl/numpyro/issues/1461).

### Confusing static behavior for members of classes.

This is how it works:

- the object id is used as the hash.
- every member of the object is treated as also static.
- but because the object id is the top level hash, the members do not need to be hashed and are just treated as compile-time constant.
- this means that if you later changes the members of the class, JAX will not reflect those changes!

Demonstration:

```
import jax
from functools import partial

class wow:
    def __init__(self, x):
        self.x = x

    @partial(jax.jit, static_argnums=(0,))
    def f(self):
        return self.x
W = wow(10)
print(W.f())
W.x = 9
print(W.f())
```

output:

```
10
10
```

### Links that go into deep JAX details:

- [Hashing a Jax.DeviceArray](https://github.com/google/jax/issues/4572#issuecomment-709809897)
- [PartialVal is a hidden internal API for caching intermediate computations in JAX](https://github.com/google/jax/discussions/9778)
- [reverse-mode differentiable bounded while loop](https://github.com/patrick-kidger/diffrax/blob/2b4e4d863c15abc7143919bac7825090bbfe50be/diffrax/misc/bounded_while_loop.py)
- ["why is jax so fast?"](https://github.com/google/jax/discussions/11078)
- [can you precompile a JAX function before running it the first time?](https://github.com/google/jax/discussions/11600)
  - yes, and maybe you should do it inside a separate thread so that the main thread can continue doing whatever it is doing!
- [conditionals based on lax.cond will evaluate lazily, while conditionals based on lax.switch will evaluate every branch regardless of the condition.](https://github.com/google/jax/discussions/11153)
- [A small library for creating and manipulating custom JAX Pytree classes](https://cgarciae.github.io/treeo/)
- [fast lookup tables in jax](https://github.com/google/jax/discussions/10475)
- [np.interp in jax](https://github.com/google/jax/issues/3860) - note that we also have an interpnd implementation!
- [scan vs while_loop](https://github.com/google/jax/discussions/3850)
- [higher order derivatives via taylor series!](https://jax.readthedocs.io/en/latest/jax.experimental.jet.html)
- [extending jax with a custom C++ or CUDA operation](https://github.com/dfm/extending-jax)
- [experimental sparse support](https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html)
- [an issue about implementing scipy.spatial](https://github.com/google/jax/issues/9235)
  - An interesting question: Would it be possible to implement a JAX KDTree?? What restrictions would make it possible? What modifications could be made so that it works well?
