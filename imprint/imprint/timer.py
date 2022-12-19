"""
This module makes it easy to mock the non-repeating timer function used by
imprint so that test results can be made reproducible.

non-repeating: The timer function will never return the same value twice even
if the time has not changed.

To replace the timer with the incrementing mock, use the following code:

with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
    ...

or

@mock.patch("imprint.timer._timer", ip.timer.new_mock_timer())
def test_something():
    ...

NOTE: The reason we mock the _timer function instead of the timer function is
because the timer function will have already been imported by calling modules
by the time the mock is applied. This means that the mock would not be applied.
"""
import time

import numpy as np


class Timer:
    def __init__(self):
        self.last = None

    def __call__(self):
        t = np.uint64(int(time.time()))
        if self.last is not None and t <= self.last:
            t = self.last + 1
        self.last = t
        return t


_timer = Timer()


def new_mock_timer():
    def mock_timer():
        mock_timer.i += 1
        return np.uint64(mock_timer.i - 1)

    mock_timer.i = 0
    return mock_timer


def timer():
    return _timer()
