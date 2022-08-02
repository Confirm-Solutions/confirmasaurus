# Developing on GPUs

This is just a place to put tips and advice for working with our tools on GPUs.


### I'm getting weird errors about cuBlas execution failing

Like:

```
jax._src.traceback_util.UnfilteredStackTrace: jaxlib.xla_extension.XlaRuntimeError: INTERNAL: CustomCall failed: jaxlib/cuda/cublas_kernels.cc:46: operation cublasCreate(&handle) failed: cuBlas has not been initialized
```

```
jax._src.traceback_util.UnfilteredStackTrace: jaxlib.xla_extension.XlaRuntimeError: INTERNAL: CustomCall failed: jaxlib/cuda/cublas_kernels.cc:119: operation cublasDtrsmBatched( handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha, const_cast<const double**>(a_batch_ptrs), lda, b_batch_ptrs, ldb, d.batch) failed: cuBlas execution failed
```

These are probably caused because you've run out of GPU memory. Check how much memory is available with `nvidia-smi`. One common cause of problems is that JAX will preallocate 90\% of GPU memory. So you can't have two GPU-using JAX processes open at the same time unless you turn off the preallocation. [Turn off preallocation by reading instructions here.](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) This is probably a useful default during development since it allows having multiple Jupyter notebooks open or a notebook and a python script.