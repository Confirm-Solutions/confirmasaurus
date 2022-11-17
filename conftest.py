from pathlib import Path

import pytest
from jax.config import config


@pytest.fixture()
def cur_loc(request):
    """The location of the file containing the current test."""
    return Path(request.fspath).parent


config.update("jax_enable_x64", True)
