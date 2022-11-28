import modal

stub = modal.Stub("example-get-started")
img = modal.Image.debian_slim().poetry_install_from_file("pyproject.toml")


@stub.function(image=img)
def f(n):
    import jax

    seed = jax.random.PRNGKey(0)
    arr = jax.random.choice(seed, n, shape=(n, n))
    return arr.sum()


if __name__ == "__main__":
    with stub.run():
        # the square of 42 is 1764
        print(f(10))
