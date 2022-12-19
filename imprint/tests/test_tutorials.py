import time
from pathlib import Path

import IPython
import matplotlib


def run_tutorial(filename):
    matplotlib.use("Agg")
    ipy = IPython.terminal.embed.InteractiveShellEmbed()
    path = Path(__file__).resolve().parent.parent.joinpath("tutorials", filename)
    start = time.time()
    ipy.run_line_magic("run", str(path))
    end = time.time()
    return ipy.user_ns, end - start


def test_ztest(snapshot):
    nb_namespace, _ = run_tutorial("ztest.ipynb")
    snapshot(nb_namespace["rej_df"])
