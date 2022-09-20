def C6(n, p):
    return (
        n
        * p
        * (1 - p)
        * (
            1
            - 30 * p * (1 - p) * (1 - 4 * p * (1 - p))
            + 5 * n * p * (1 - p) * (5 - 26 * p * (1 - p))
            + 15 * n**2 * p**2 * (1 - p) ** 2
        )
    )


# todo: test against c6, maybe also c5 from wikipedia?
# def test_odi_constant():

# todo: test the 0.5 p crossing
# todo: test the second holdering?
