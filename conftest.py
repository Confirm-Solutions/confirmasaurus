from pathlib import Path

import pytest


@pytest.fixture()
def cur_loc(request):
    return Path(request.fspath).parent
