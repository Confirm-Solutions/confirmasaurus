import numpy as np


def pad_arg__(a, axis, n_pad: int):
    pad_element = np.take(a, indices=0, axis=axis)
    pad_element = np.expand_dims(pad_element, axis=axis)
    new_shape = tuple(a.shape[i] if i != axis else n_pad for i in range(a.ndim))
    return np.concatenate((a, np.full(new_shape, pad_element)), axis=axis)


def create_batched_args__(args, in_axes, start, end, n_pad=None):
    def arg_transform(arg, axis):
        return pad_arg__(arg, axis, n_pad) if n_pad is not None else arg

    return [
        arg_transform(
            np.take(arg, indices=range(start, end), axis=axis),
            axis,
        )
        if axis is not None
        else arg
        for arg, axis in zip(args, in_axes)
    ]


def batch(f, batch_size: int, in_axes):
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

        dim = dims[0]
        batch_size_new = min(batch_size, dim)
        n_full_batches = dim // batch_size_new
        remainder = dim % batch_size_new
        n_pad = batch_size_new - remainder
        pad_last = remainder > 0
        start = 0
        end = batch_size_new

        for _ in range(n_full_batches):
            batched_args = create_batched_args__(
                args=args,
                in_axes=in_axes,
                start=start,
                end=end,
            )
            yield (f(*batched_args), 0)
            start += batch_size_new
            end += batch_size_new

        if pad_last:
            batched_args = create_batched_args__(
                args=args,
                in_axes=in_axes,
                start=start,
                end=dim,
                n_pad=n_pad,
            )
            yield (f(*batched_args), n_pad)

    return internal


def batch_all(f, batch_size: int, in_axes):
    f_batch = batch(f, batch_size, in_axes)

    def internal(*args):
        outs = tuple(out for out in f_batch(*args))
        return tuple(out[0] for out in outs), outs[-1][-1]

    return internal
