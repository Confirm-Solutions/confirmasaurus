import modal

stub = modal.Stub("confirm")

img = modal.Image.from_dockerhub(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    setup_commands=[
        "apt-get update",
        "apt-get install -y python-is-python3 python3-pip",
    ],
).poetry_install_from_file("pyproject.toml", "poetry.lock")


@stub.function(image=img, gpu=True)
def f(n):
    import jax

    seed = jax.random.PRNGKey(0)
    arr = jax.random.choice(seed, n, shape=(n, n))
    return float(arr.sum())


if __name__ == "__main__":
    with stub.run():
        # outputs 427
        for i in range(10):
            print(f(10))
