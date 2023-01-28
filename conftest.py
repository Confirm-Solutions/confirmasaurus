from pathlib import Path

import pytest
from jax.config import config

config.update("jax_enable_x64", True)


@pytest.fixture()
def cur_loc(request):
    """The location of the file containing the current test."""
    return Path(request.fspath).parent


# NOTE: The run-modal-safe and run-modal-unsafe flags might not be necessary in
# the future if Modal implements Modal-inside-Modal. This seems very likely!
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "modal_unsafe: mark test as unsafe to run on Modal."
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip tests based on --run-modal-safe and --run-modal-unsafe flags.
    """
    if config.getoption("--run-modal-safe"):
        skip_modal_unsafe = pytest.mark.skip(
            reason="--run-modal-safe prevented running this test"
        )
        for item in items:
            if "modal_unsafe" in item.keywords:
                item.add_marker(skip_modal_unsafe)
    elif config.getoption("--run-modal-unsafe"):
        skip_modal_safe = pytest.mark.skip(
            reason="--run-modal-unsafe prevented running this test"
        )
        for item in items:
            if "modal_unsafe" not in item.keywords:
                item.add_marker(skip_modal_safe)
    else:
        return


def pytest_addoption(parser):
    """ """
    parser.addoption(
        "--run-modal-safe",
        action="store_true",
        default=False,
        help="run only Modal-safe tests",
    )
    parser.addoption(
        "--run-modal-unsafe",
        action="store_true",
        default=False,
        help="run only Modal-unsafe tests",
    )
