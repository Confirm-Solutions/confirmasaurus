import jax.numpy as jnp


class ArraySlice0:
    def __init__(self, a, start, end):
        self.array = a
        self.start = start
        self.end = end  # TODO: unused

    def __getitem__(self, index):
        return self.array[self.start + index]


class ArrayReshape0:
    def __init__(self, a, shape):
        self.array = a
        self.shape = shape
        self.mask = jnp.flip(jnp.cumprod(jnp.flip(self.shape[1:])))

    def __getitem__(self, index):
        i = index[-1] + jnp.sum(self.mask * index[:-1])
        return self.array[i]


def slice0(a, start, end):
    """
    Slices an array along axis 0 from start and end.

    Parameters:
    -----------
    a:          array to slice along axis 0.
    start:      starting position to slice.
    end:        ending position to slice (non-inclusive).
    """
    return ArraySlice0(a, start, end)


def reshape0(a, shape):
    """
    Reshapes a given array along the 0th axis
    with a new shape.

    Parameters:
    -----------
    a:          array to reshape along axis 0.
    shape:      new shape of array along axis 0.
    """
    return ArrayReshape0(a, shape)
