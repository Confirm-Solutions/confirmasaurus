import modal

stub = modal.Stub("example-get-started")
img = modal.Image.debian_slim().poetry_install_from_file("pyproject.toml")


@stub.function(image=img)
def f(n):
    import pkg_resources

    return pkg_resources.get_distribution("poetry").version


if __name__ == "__main__":
    with stub.run():
        # the square of 42 is 1764
        print(f(10))
