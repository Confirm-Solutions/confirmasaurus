"""
The driver script for the 4D berry code that ran to produce our 2022 paper.
"""
import typer

import confirm.berrylib.batch_run as batch_run

app = typer.Typer()


@app.command()
def onebatch(begin: int = 0, end: int = None):
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


@app.command()
def main():
    for i in range(4, 17):
        print(i, i + 1)
        size = 1000000
        begin = i * size
        end = (i + 1) * size
        onebatch(begin, end)


if __name__ == "__main__":
    app()
