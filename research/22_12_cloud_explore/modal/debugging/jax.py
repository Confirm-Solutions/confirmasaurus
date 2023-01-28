import modal

stub = modal.Stub("jaxtest")

# Option #1: build off of tensorflow/tensorflow
# img = (
#     modal.Image.from_dockerhub('tensorflow/tensorflow:latest-gpu')
#     .pip_install(
#         ["jax[cuda]"],
#         "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
#     )
# )

# Option #2: build off of nvidia/cuda
img = modal.Image.from_dockerhub(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    setup_commands=[
        "apt-get update",
        "apt-get install -y python-is-python3 python3-pip",
    ],
).pip_install(
    ["jax[cuda]"], "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
)


@stub.function(image=img, gpu=True)
def f(n):
    import jax

    seed = jax.random.PRNGKey(0)
    arr = jax.random.choice(seed, n, shape=(n, n))
    return arr.sum()


if __name__ == "__main__":
    with stub.run():
        # outputs 427
        print(f(10))
