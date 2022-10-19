import jax.numpy as jnp
import numpy as np


def _pad_arg(a, axis, n_pad: int):
    """
    Pads an array:
    - along the specified axis.
    - with the values at index 0
    - by n_pad elements.

    Padding with the values at index 0 avoids problems with using a placeholder
    value like 0 in situations where the placeholder value would be invalid.
    """
    pad_element = np.take(a, indices=0, axis=axis)
    pad_element = np.expand_dims(pad_element, axis=axis)
    new_shape = tuple(a.shape[i] if i != axis else n_pad for i in range(a.ndim))
    return np.concatenate((a, np.full(new_shape, pad_element)), axis=axis)


def _create_batched_args(args, in_axes, start, end, n_pad=None):
    """
    Subsets and pads the arguments as specified in in_axes.
    """

    def arg_transform(arg, axis):
        return _pad_arg(arg, axis, n_pad) if n_pad is not None else arg

    return [
        arg_transform(
            np.take(arg, indices=range(start, end), axis=axis),
            axis,
        )
        if axis is not None
        else arg
        for arg, axis in zip(args, in_axes)
    ]


def batch_yield(f, batch_size: int, in_axes):
    """
    A generator that yields batches of output from the function f.

    Args:
        f: The function to be batched.
        batch_size: The batch size.
        in_axes: For each argument, the axis along which to batch. If None, the
            argument is not batched.
    """

    def internal(*args):
        dims = np.array(
            [arg.shape[axis] for arg, axis in zip(args, in_axes) if axis is not None]
        )
        if len(dims) <= 0:
            raise ValueError(
                "f must take at least one argument "
                "whose corresponding in_axes is not None."
            )

        dims_all_equal = np.sum(dims != dims[0]) == 0
        if not dims_all_equal:
            raise ValueError(
                "All batched arguments must have the same dimension "
                "along their corresopnding in_axes."
            )

        if len(args) != len(in_axes):
            raise ValueError(
                "The number of arguments must match the number of in_axes."
            )

        dim = dims[0]

        # NOTE: i don't think we should shrink the batch size because that'll
        # incur extra JIT overhead when a user calls with lots of different
        # small sizes. but we could make this a configurable behavior.
        # batch_size_new = min(batch_size, dim)
        n_full_batches = dim // batch_size
        remainder = dim % batch_size
        n_pad = batch_size - remainder
        pad_last = remainder > 0
        start = 0
        end = batch_size

        for _ in range(n_full_batches):
            batched_args = _create_batched_args(
                args=args,
                in_axes=in_axes,
                start=start,
                end=end,
            )
            # TODO: why return n_pad?
            yield (f(*batched_args), 0)
            start += batch_size
            end += batch_size

        if pad_last:
            batched_args = _create_batched_args(
                args=args,
                in_axes=in_axes,
                start=start,
                end=dim,
                n_pad=n_pad,
            )
            yield (f(*batched_args), n_pad)

    return internal


def batch_all(f, batch_size: int, in_axes):
    """
    A function wrapper that batches calls to f.

    Args:
        f: Function to be batched.
        batch_size: The batch size.
        in_axes: For each argument, the axis along which to batch. If None, the
            argument is not batched.

    Returns:
        The batched results.
    """
    f_batch = batch_yield(f, batch_size, in_axes)

    def internal(*args):
        outs = tuple(out for out in f_batch(*args))
        return tuple(out[0] for out in outs), outs[-1][-1]

    return internal


def batch(f, batch_size: int, in_axes, out_axes=None):
    """
    Batch a function call and concatenate the output.

    The API is intended to be similar to jax.vmap.
    https://jax.readthedocs.io/en/latest/_modules/jax/_src/api.html#vmap

    If the function has a single output, the output is concatenated along the
    specified axis. If the function has multiple outputs, each output is
    concatenated along the corresponding axis.

    Args:
        f: Function to be batched.
        batch_size: The batch size.
        in_axes: For each argument, the axis along which to batch. If None, the
            argument is not batched.
        out_axes: The axis along which to concatenate function outputs.
            Defaults to None which will concatenate along the first axis.

    Returns:
        A concatenated array or a tuple of concatenated arrays.
    """
    f_batch_all = batch_all(f, batch_size, in_axes)

    def internal(*args):
        outs, n_pad = f_batch_all(*args)
        internal_out_axes = (
            out_axes if out_axes is not None else tuple(0 for _ in range(len(outs[0])))
        )
        last_j = len(outs) - 1

        return_first = False
        if isinstance(outs[0], np.ndarray) or isinstance(outs[0], jnp.DeviceArray):
            return_first = True
            outs = [[o] for o in outs]

        return_vals = [
            np.concatenate(
                [
                    outs[j][i][:-n_pad] if (j == last_j and n_pad != 0) else outs[j][i]
                    for j in range(len(outs))
                ],
                axis=internal_out_axes[i],
            )
            for i in range(len(outs[0]))
        ]
        if return_first:
            return return_vals[0]
        else:
            return return_vals

    return internal
