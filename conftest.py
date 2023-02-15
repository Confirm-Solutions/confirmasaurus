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
