```python
import coiled

coiled.create_software_environment(
    name="smalldev",
    container="ghcr.io/confirm-solutions/smalldev:latest",
)
```

```python
cluster = coiled.Cluster(
    software="smalldev",
    n_workers=2,
    scheduler_gpu=True,  # recommended
    worker_gpu=1,  # single T4 per worker
    worker_class="dask_cuda.CUDAWorker",  # recommended
    environ={"DISABLE_JUPYTER": "true"},  # needed for "stable" RAPIDS image
)
```

```python
import coiled
# create a remote Dask cluster with Coiled
cluster = coiled.Cluster(name="my-cluster", n_workers=2)
```

```python
# connect a Dask client to the cluster
client = cluster.get_client()

# link to Dask scheduler dashboard
print("Dask scheduler dashboard:", client.dashboard_link)

```

```python
import dask

@dask.delayed
def run_ada():
    import jax
    import jax.numpy as jnp
    seed = jax.random.PRNGKey(0)
    data = jax.random.uniform(seed, shape=(100000, 2))
    return jnp.sum(data[:,0] > data[:,1])
    # import numpy as np
    # import imprint as ip
    # from imprint.models.ztest import ZTest1D
    # import confirm.adagrid as ada
    # d = 1
    # g = ip.cartesian_grid(
    #     theta_min=np.full(d, -1),
    #     theta_max=np.full(d, 0),
    #     null_hypos=[ip.hypo("theta0 > theta1")],
    # )
    # db = ada.ada_validate(
    #     ZTest1D,
    #     g=g,
    #     lam=-1.96,
    #     prod=False
    # )
    # return ip.Grid(db.get_results(), 1).prune_inactive().n_tiles

NN = run_ada().compute()
```

```python
df.shape[0]
```

```python
client.close()
```

```python
cluster.close()
```
