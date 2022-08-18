import numpy as np


# TODO: warning. this code has a bug and i don't know exactly where.this code
# is still sitting here despite not being used because I think it's going to be
# very useful in the future.
def batch_execute(f, batch_dim, batch_size, *args):
    """
    batch:
        The batched execution mode runs chunks of a fixed number of
        inferences. Chunking and padding is a useful technique to avoid
        recompilation when calling FullLaplace multiple times with
        different data sizes. Each call to _call_backend results in a JAX
        JIT operation. These can be slow, especially in comparison to small
        amounts of data. Chunking is also useful for very large numbers of
        datasets because it relieves memory pressure and is often faster
        due to better use of caches

        The non-batched execution mode can sometimes be faster but incurs a
        substantial JIT cost when recompiling for each different input
        shape.
    """
    """
    TODO: it would be feasible to re-design this to have a similar interface
    to jax.vmap

    An interesting jax problem is that many jit-ted JAX functions require
    re-compiling every time you change the array size of the inputs. There’s no
    general solution to this and if you’re running the same operation on a bunch of
    differently sized arrays, then the compilation time can become really annoying
    and expensive.

    One solution is to pad and batch your inputs. For example, we might decide
    to only run a function on chunks of 2^16 values. Then, any smaller set of
    values is padded with zeros out to 2^16. Any larger set of values is
    batched into a bunch of function calls. This has negative side effects
    (more complexity, potential performance consequences) but it succeeds in
    the goal of only ever compiling the function once.  It also often has the
    positive side effect of reducing memory usage.
    """
    n_batchs = int(np.ceil(batch_dim / batch_size))
    pad_N = batch_size * n_batchs

    def leftpad(arr):
        pad_spec = [[0, 0] for dim in arr.shape]
        rem = batch_dim % batch_size
        pad_spec[0][1] = batch_size - (batch_size if rem == 0 else rem)
        out = np.pad(arr, pad_spec)
        assert out.shape[0] == pad_N
        return out

    pad_args = [leftpad(a) if should_batch else a for a, should_batch in args]
    pad_out = None
    for i in range(n_batchs):
        start = i * batch_size
        end = (i + 1) * batch_size
        args_batch = [
            pad_args[i][start:end] if should_batch else pad_args[i]
            for i, (_, should_batch) in enumerate(args)
        ]
        out_batch = f(*args_batch)
        if pad_out is None:
            pad_out = []
            for arr in out_batch:
                shape = list(arr.shape)
                if len(shape) == 0:
                    shape = [pad_N]
                shape[0] = pad_N
                pad_out.append(np.empty_like(arr, shape=shape))
        for target, source in zip(pad_out, out_batch):
            target[start:end] = source
    return [arr[:batch_dim] for arr in pad_out]
