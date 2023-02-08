from pathlib import Path

import dotenv
import pytest
from jax.config import config

config.update("jax_enable_x64", True)
dotenv.load_dotenv()


@pytest.fixture()
def cur_loc(request):
    """The location of the file containing the current test."""
    return Path(request.fspath).parent
