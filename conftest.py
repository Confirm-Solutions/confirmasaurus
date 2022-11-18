from pathlib import Path

import pytest
from jax.config import config

from confirm.mini_imprint.testing import Pickler  # noqa
from confirm.mini_imprint.testing import pytest_addoption  # noqa
from confirm.mini_imprint.testing import snapshot  # noqa
from confirm.mini_imprint.testing import TextSerializer  # noqa

config.update("jax_enable_x64", True)


@pytest.fixture()
def cur_loc(request):
    """The location of the file containing the current test."""
    return Path(request.fspath).parent
