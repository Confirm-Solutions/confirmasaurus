import numpy as np


def make_cartesian_grid_range(size, lower, upper):
    import pyimprint.grid as pygrid

    assert lower.shape[0] == upper.shape[0]

    # make initial 1d grid
    center_grids = (
        pygrid.Gridder.make_grid(size, lower[i], upper[i]) for i in range(len(lower))
    )

    # make a grid of centers
    coords = np.meshgrid(*center_grids)
    centers = np.concatenate([c.flatten().reshape(-1, 1) for c in coords], axis=1)

    # make corresponding radius
    radius = np.array(
        [pygrid.Gridder.radius(size, lower[i], upper[i]) for i in range(len(lower))]
    )
    radii = np.full(shape=centers.shape, fill_value=radius)

    return centers, radii
