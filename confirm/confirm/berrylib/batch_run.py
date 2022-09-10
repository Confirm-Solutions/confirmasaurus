import logging
import sys
import time

import numpy as np
import typer
from scipy.special import logit

import confirm.berrylib.fast_inla as fast_inla
import confirm.mini_imprint.binomial as binomial
import confirm.mini_imprint.grid as grid


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.handlers = []
    root.addHandler(handler)
    logging.getLogger("absl").setLevel(logging.WARN)


logger = logging.getLogger(__name__)

# TODO add tests for the batching to check equality with unbatched?


def build_grid(config):
    n_arms = config["n_arms"]

    logger.info("setting up grid.")
    null_hypos = [
        grid.HyperPlane(-np.identity(n_arms)[i], -logit(0.1)) for i in range(n_arms)
    ]
    theta1d = [
        np.linspace(
            config["theta_min"], config["theta_max"], 2 * config["n_theta_1d"] + 1
        )[1::2]
        for _ in range(n_arms)
    ]
    theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
    radii = np.empty(theta.shape)
    for i in range(theta.shape[1]):
        radii[:, i] = 0.5 * (theta1d[i][1] - theta1d[i][0])

    # filter to only the specified subset of the grid.
    begin = config["gridpt_batch_begin"]
    end = config["gridpt_batch_end"]
    end = min(theta.shape[0], end if end is not None else theta.shape[0])
    logger.info(f"processing only grid points {begin}:{end}")
    theta = theta[begin:end]
    radii = radii[begin:end]

    g = grid.prune(grid.build_grid(theta, radii, null_hypos))
    theta = g.thetas
    theta_tiles = g.thetas[g.grid_pt_idx]

    tile_radii = g.radii[g.grid_pt_idx]
    corners = g.vertices
    return (theta, theta_tiles, tile_radii, corners, g.null_truth), begin, end


def accumulate(config, grid_output):
    _, theta_tiles, _, _, null_truth = grid_output

    n_arms = config["n_arms"]

    logger.info("building rejection table")
    fi = fast_inla.FastINLA(n_arms=n_arms)
    rejection_table = binomial.build_rejection_table(
        n_arms, config["n_arm_samples"], fi.rejection_inference
    )

    np.random.seed(config["seed"])
    accumulator = binomial.binomial_accumulator(
        lambda data: binomial.lookup_rejection(rejection_table, data[..., 0])
    )

    # batching improves performance dramatically for larger tile counts:
    # ~6x for a 64^3 grid
    typeI_sum = np.zeros(theta_tiles.shape[0])
    typeI_score = np.zeros((theta_tiles.shape[0], n_arms))
    n_gridpt_batches = int(np.ceil(theta_tiles.shape[0] / config["gridpt_batch_size"]))
    n_sim_batches = config["sim_size"] // config["sim_batch_size"]

    logger.info(
        "running accumulation with: "
        f" n_tiles={theta_tiles.shape[0]} "
        f" batches (sim={n_sim_batches}, grid = {n_gridpt_batches})"
    )
    assert config["sim_size"] % config["sim_batch_size"] == 0
    for j in range(n_sim_batches):
        samples = np.random.uniform(
            size=(config["sim_batch_size"], config["n_arm_samples"], n_arms)
        )
        logger.info(f"beginning sim batch {j}")
        start = time.time()
        for i in range(0, n_gridpt_batches):
            gridpt_start = i * config["gridpt_batch_size"]
            gridpt_end = (i + 1) * config["gridpt_batch_size"]
            gridpt_end = min(gridpt_end, theta_tiles.shape[0])
            sum_batch, score_batch = accumulator(
                theta_tiles[gridpt_start:gridpt_end],
                null_truth[gridpt_start:gridpt_end],
                samples,
            )
            typeI_sum[gridpt_start:gridpt_end] += sum_batch
            typeI_score[gridpt_start:gridpt_end] += score_batch
        logger.info(f"finished sim batch, took {time.time() - start:.2f}s")

    sim_sizes = np.full(theta_tiles.shape[0], config["sim_size"])

    return (sim_sizes, typeI_sum, typeI_score)


def build_upper_bounds(config, grid_output, sim_output):
    _, theta_tiles, tile_radii, corners, _ = grid_output
    sim_sizes, typeI_sum, typeI_score = sim_output

    logger.info("computing upper bounds.")
    total, d0, d0u, d1w, d1uw, d2uw = binomial.upper_bound(
        theta_tiles,
        tile_radii,
        corners,
        sim_sizes,
        config["n_arm_samples"],
        typeI_sum,
        typeI_score,
    )

    return (total, d0, d0u, d1w, d1uw, d2uw)


app = typer.Typer()


@app.command()
def cli_main(
    n_arm_samples: int = 35,
    seed: int = 10,
    name: str = "berry3d",
    n_arms: int = 3,
    n_theta_1d: int = 24,
    sim_size: int = 50000,
    theta_min: float = -2.8,
    theta_max: float = -1.2,
    gridpt_batch_size: int = 5000,
    sim_batch_size: int = 50000,
    gridpt_batch_begin: int = 0,
    gridpt_batch_end: int = None,
):
    main(locals())


def main(config):
    from rich import print as rich_print

    rich_print("Running Berry model with config:")
    rich_print(config)
    setup_logging()

    grid_output, begin, end = build_grid(config)
    sim_output = accumulate(config, grid_output)
    bound_output = build_upper_bounds(config, grid_output, sim_output)

    out_filename = f"output_{config['name']}_{begin}_{end}.npy"
    logger.info(f"saving output to {out_filename}")
    np.save(
        out_filename,
        np.array([grid_output, sim_output, bound_output], dtype=object),
    )


if __name__ == "__main__":
    app()
