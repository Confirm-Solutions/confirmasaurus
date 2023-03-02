import sys
from pathlib import Path

import dotenv
import pandas as pd
import pytest
from jax.config import config

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 500)

config.update("jax_enable_x64", True)
dotenv.load_dotenv()


@pytest.fixture()
def cur_loc(request):
    """The location of the file containing the current test."""
    return Path(request.fspath).parent


def pytest_collection_modifyitems(items):
    """
    In March 2023, the time to import confirm was rising towards 2.5 or 3
    seconds. That sucks for fast iteration. In order to fix this, I went through
    and identified packages that are used only occasionally and moved those
    imports inside the functions that use them. This lazy importing got
    import time down to 0.9 seconds.

    Lazy importing violates python convention, but fast startup is too
    important to give up. In some ways, lazy imports are also a good incentive
    to isolate dependencies to narrow portions of the codebase which will have
    positive side effects.

    There are a few slow-to-import packages that are used too much to be lazily
    imported: pandas, jax, jax.numpy, numpy.

    A useful benchmark is to run:

    ```
    time python -X importtime -m pytest -k "hihihihi
    ```

    On my system, this is currently running in 0.9 seconds when run warm.
    """
    for m in [
        "sympy",
        "matplotlib",
        "IPython",
        "scipy",
        "numpyro",
        "jax.scipy.special",
        "duckdb",
        "clickhouse_connect",
    ]:
        if m in sys.modules:
            raise Exception(
                f"Module {m} must be imported lazily. See the docstring in conftest.py."
            )
