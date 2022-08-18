# JAX spits out a bunch of ugly warnings on import. Ignore them.
# https://github.com/google/flatbuffers/issues/6957
# DeprecationWarning: the imp module is deprecated in favour of importlib; see
# the module's documentation for alternative uses
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jax

# Also, we'll centrally enable 64-bit JAX. This keeps other files less
# cluttered.
# TODO: there should be a better way to do this that allows the user to specify
# whether they want to support 64 bit or not. Enabling 64 bit seems to make a
# number of operations substantially slower!
from jax.config import config

config.update("jax_enable_x64", True)
