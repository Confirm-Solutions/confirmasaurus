import typer

import confirm.berrylib.batch_run as batch_run

app = typer.Typer()


@app.command()
def main(begin: int = 0, end: int = None):
    config = dict(
        n_arm_samples=35,
        seed=10,
        name="berry4d",
        n_arms=4,
        n_theta_1d=64,
        sim_size=500000,
        theta_min=-3.5,
        theta_max=1.0,
        gridpt_batch_size=10000,
        sim_batch_size=10000,
        gridpt_batch_begin=begin,
        gridpt_batch_end=end,
    )
    batch_run.main(config)


if __name__ == "__main__":
    app()
