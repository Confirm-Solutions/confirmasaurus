import time

import numpy as np
import pytest

import confirm.mini_imprint.grid as grid


def normalize(n):
    return n / np.linalg.norm(n)


@pytest.fixture
def simple_grid():
    thetas = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    hypos = [
        grid.HyperPlane(normalize(np.array([1, -1])), 0),
        grid.HyperPlane(normalize(np.array([1, 1])), -1),
    ]
    return grid.build_grid(thetas, radii, hypos, should_prune=False)


def test_cartesian_gridpts():
    theta, radii = grid.cartesian_gridpts([-1, -1], [1, 1], [2, 2])
    g = grid.build_grid(theta, radii)
    assert np.all(g.grid_pt_idx == [0, 1, 2, 3])
    assert g.null_truth.shape[1] == 0

    null_hypos = [grid.HyperPlane(-np.identity(2)[i], -0.1) for i in range(2)]
    g = grid.build_grid(theta, radii, null_hypos)
    assert np.all(g.grid_pt_idx == [0, 1, 1, 2, 2, 3, 3, 3, 3])


def test_tile_copy(simple_grid):
    g = simple_grid
    np.testing.assert_allclose(g.grid_pt_idx, [1, 2, 3, 3, 0, 0, 0, 0])
    np.testing.assert_allclose(
        g.null_truth,
        np.array([[0, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [0, 1], [0, 0]]),
    )


def test_tile_prune(simple_grid):
    g = simple_grid
    gp = grid.prune(g)
    np.testing.assert_allclose(gp.grid_pt_idx, [1, 2, 3, 3, 0, 0, 0])
    np.testing.assert_allclose(
        gp.null_truth,
        np.array([[0, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [0, 1]]),
    )


def test_prune_off_gridpt():
    thetas = np.array([[-0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    hypos = [grid.HyperPlane(normalize(np.array([1, 1])), 0)]
    g = grid.prune(grid.build_grid(thetas, radii, hypos))
    np.testing.assert_allclose(g.thetas, np.array([[0.5, 0.5]]))
    np.testing.assert_allclose(g.grid_pt_idx, np.array([0]))


def test_prune_is_regular():
    thetas = np.array([[0.0, 0.0]])
    radii = np.full_like(thetas, 0.5)
    hypos = [grid.HyperPlane(normalize(np.array([1, 1])), 0)]
    g = grid.build_grid(thetas, radii, hypos, should_prune=False)
    # np.testing.assert_allclose(g.thetas, np.array([[0.0, 0.0]]))
    np.testing.assert_allclose(g.grid_pt_idx, np.array([0, 0]))
    gp = grid.prune(g)
    np.testing.assert_allclose(gp.grid_pt_idx, np.array([0]))


def test_prune_no_surfaces():
    thetas = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radii = np.full_like(thetas, 0.5)
    g = grid.build_grid(thetas, radii, [])
    gp = grid.prune(g)
    assert g == gp


def test_prune_twice_invariance(simple_grid):
    gp = grid.prune(simple_grid)
    gpp = grid.prune(gp)
    np.testing.assert_allclose(gp.thetas, gpp.thetas)
    np.testing.assert_allclose(gp.radii, gpp.radii)
    np.testing.assert_allclose(gp.null_truth, gpp.null_truth)
    np.testing.assert_allclose(gp.grid_pt_idx, gpp.grid_pt_idx)


def test_refine():
    n_arms = 2
    theta, radii = grid.cartesian_gridpts(
        np.full(n_arms, -3.0), np.full(n_arms, 1.0), np.full(n_arms, 4)
    )
    null_hypos = [grid.HyperPlane(-np.identity(n_arms)[i], 1.1) for i in range(n_arms)]
    g = grid.prune(grid.build_grid(theta, radii, null_hypos))
    refine_tiles = np.array([0, 3, 4, 5])
    refine_gridpts = g.grid_pt_idx[refine_tiles]
    new_theta, new_radii, keep_tiles = grid.refine_grid(g, refine_gridpts)
    np.testing.assert_allclose(
        keep_tiles, np.array([1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    )
    # np.testing.assert_allclose(g.grid_pt_idx[keep_tiles], unrefined.grid_pt_idx)
    np.testing.assert_allclose(new_radii, 0.25)

    pts_to_refine = np.array([[-2.5, -2.5], [-2.5, -0.5], [-2.5, 0.5], [-1.5, -2.5]])
    for i in range(2):
        for j in range(2):
            subset = new_theta[(2 * i + j) :: 4]
            correct = pts_to_refine - np.array([2 * i - 1, 2 * j - 1]) * 0.25
            np.testing.assert_allclose(subset, correct)


def test_prune_and_symmetry():
    null_hypos = [
        grid.HyperPlane([1, -1, 0, 0], 0),
        grid.HyperPlane([1, 0, -1, 0], 0),
        grid.HyperPlane([1, 0, 0, -1], 0),
    ]
    syms = [grid.HyperPlane([0, 1, -1, 0], 0), grid.HyperPlane([0, 0, 1, -1], 0)]

    theta, radii = grid.cartesian_gridpts(np.full(4, -1), np.full(4, 1), np.full(4, 4))

    g = grid.build_grid(
        theta, radii, null_hypos=null_hypos, symmetry_planes=syms, should_prune=True
    )

    Hns = np.array([H.n for H in null_hypos])
    is_alt = np.all(g.theta_tiles.dot(Hns.T) < 0, axis=1)
    assert not np.any(is_alt)
    assert np.all(g.theta_tiles[:, 1] >= g.theta_tiles[:, 2])
    assert np.all(g.theta_tiles[:, 2] >= g.theta_tiles[:, 3])
    assert g.n_tiles == 133

    g2 = grid.build_grid(theta, radii, null_hypos=null_hypos)
    Hns = np.array([H.n for H in null_hypos])
    is_alt = np.all(g2.theta_tiles.dot(Hns.T) < 0, axis=1)
    is_sym = (g2.theta_tiles[:, 1] < g2.theta_tiles[:, 2]) | (
        g2.theta_tiles[:, 2] < g2.theta_tiles[:, 3]
    )
    assert (g2.n_tiles - np.sum(is_alt | is_sym)) == g.n_tiles


n_arms = 4
n_theta_1d = 52


def py_grid():
    null_hypos = [grid.HyperPlane(-np.identity(n_arms)[i], 2) for i in range(n_arms)]
    theta1d = [np.linspace(-3.5, 1.0, 2 * n_theta_1d + 1)[1::2] for i in range(n_arms)]
    theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
    radii = np.empty(theta.shape)
    for i in range(theta.shape[1]):
        radii[:, i] = 0.5 * (theta1d[i][1] - theta1d[i][0])
    g = grid.prune(grid.build_grid(theta, radii, null_hypos))
    return g


def cpp_grid():
    import pyimprint.grid as grid

    # define null hypos
    null_hypos = []
    for i in range(n_arms):
        n = np.zeros(n_arms)
        # null is:
        # theta_i <= logit(0.1)
        # the normal should point towards the negative direction. but that also
        # means we need to negate the logit(0.1) offset
        n[i] = -1
        null_hypos.append(grid.HyperPlane(n, 2))

    gr = grid.make_cartesian_grid_range(
        n_theta_1d, np.full(n_arms, -3.5), np.full(n_arms, 1.0), 1
    )
    gr.create_tiles(null_hypos)
    gr.prune()
    return gr


def benchmark(f, iter=3):
    runtimes = []
    for i in range(iter):
        start = time.time()
        f()
        end = time.time()
        runtimes.append(end - start)
    return runtimes


if __name__ == "__main__":
    # Runtimes:
    # py_grid: [0.4465768337249756, 0.40291690826416016, 0.3762087821960449]
    # cpp_grid: [4.8662269115448, 4.6366658210754395, 4.647104024887085]6
    print("py_grid:", benchmark(py_grid, iter=1))
    print("cpp_grid:", benchmark(cpp_grid))
