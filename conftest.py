from pathlib import Path

import pytest
from jax.config import config

from confirm.imprint.testing import Pickler  # noqa
from confirm.imprint.testing import pytest_addoption as ip_addoption  # noqa
from confirm.imprint.testing import snapshot  # noqa
from confirm.imprint.testing import TextSerializer  # noqa

config.update("jax_enable_x64", True)


def pytest_addoption(parser):
    ip_addoption(parser)
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture()
def cur_loc(request):
    """The location of the file containing the current test."""
    return Path(request.fspath).parent
