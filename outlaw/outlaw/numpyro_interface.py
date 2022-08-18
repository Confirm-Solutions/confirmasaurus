import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers


def from_numpyro(model, pin, shape):
    """
    Convert a numpyro model function to an INLA model by:
    - deriving the log joint density from the numpyro model
    - specifying the pinned parameters that will be integrated out in the INLA
      model

    Note:
        example valid inputs for pin:
        - pin = "sig2"
        - pin = ["sig2"]
        - pin = [("sig2", 0)]
        - pin = [("sig2", 0), ("theta", 1)]

    Args:
        model: a numpyro model function
        pin: a list of parameters to be fixed during INLA optimization.
            Specified as either:
            - a string, in which case that single parameter will be pinned
              during INLA optimization
            - a list of strings, in which case those parameters will be pinned
              during INLA optimization.
            - a list of tuples of the form (parameter_name, index) in which case
              the vector parameter specified by the parameter_name will be
              fixed at the specified index.
        shape: the shape of the data expected by the numpyro model

    Returns:
        A tuple containing:
        1. an example dictionary of parameters
        2. the log joint density function
    """
    d = _examine_model(model, jnp.zeros(shape))
    param_example = pin_params({k: np.zeros(n) for k, n in d.items()}, pin)

    def log_joint(p, data):
        seeded_model = handlers.seed(model, jax.random.PRNGKey(10))
        subs_model = handlers.substitute(seeded_model, p)
        trace = handlers.trace(subs_model).get_trace(data)
        return jnp.sum(
            jnp.array(
                [
                    jnp.sum(site["fn"].log_prob(site["value"]))
                    for k, site in trace.items()
                ]
            )
        )

    return log_joint, param_example


@partial(jax.jit, static_argnums=(0,))
def _examine_model(model, data):
    seeded_model = handlers.seed(model, jax.random.PRNGKey(10))
    trace = handlers.trace(seeded_model).get_trace(data)
    d = {
        k: (v["value"].shape[0] if len(v["value"].shape) > 0 else 1)
        for k, v in trace.items()
        if not v["is_observed"]
    }
    return d


def pin_params(full_params, pin):
    out = copy.deepcopy(full_params)
    if not isinstance(pin, list):
        pin = [pin]
    for pin_entry in pin:
        if isinstance(pin_entry, str):
            out[pin_entry][:] = jnp.nan
        else:
            out[pin_entry[0]][pin_entry[1]] = jnp.nan
    return out
