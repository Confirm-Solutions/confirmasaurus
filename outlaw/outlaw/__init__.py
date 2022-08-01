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
from jax.config import config

config.update("jax_enable_x64", True)
