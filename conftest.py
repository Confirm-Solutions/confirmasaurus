from pathlib import Path

import pytest


@pytest.fixture()
def cur_loc(request):
    """The location of the file containing the current test."""
    return Path(request.fspath).parent
