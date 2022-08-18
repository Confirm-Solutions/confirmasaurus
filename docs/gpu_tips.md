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

### I get an error about `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`.

This happens to me when the Codespace I've using was built a little while ago and has not been rebuilt. The problem is that the CUDA version installed inside the Codespace is older than the driver version installed outside the docker container. The solution is simply to upgrade CUDA inside the Codespace:

Solution:
```
apt update
apt upgrade
```

The full error from JAX:
```
2022-08-05 20:19:36.537524: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: system has unsupported display driver / cuda driver combination
2022-08-05 20:19:36.537838: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_diagnostics.cc:313] kernel version 515.65.1 does not match DSO version 515.48.7 -- cannot find working devices in this configuration
```