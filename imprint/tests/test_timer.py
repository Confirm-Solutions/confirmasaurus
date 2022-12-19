import time

from imprint.timer import new_mock_timer
from imprint.timer import Timer


def test_timer_zero():
    t = Timer()
    start = int(time.time())
    assert t() == start
    assert t() == start + 1
    assert t() == start + 2


def test_timer_mock():
    t = new_mock_timer()
    assert t() == 0
    assert t() == 1
    assert t() == 2
