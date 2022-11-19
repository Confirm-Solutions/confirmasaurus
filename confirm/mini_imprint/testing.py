"""
Here you will find tools for snapshot testing. Snapshot testing is a way to
check that the output of a function is the same as it used to be. This is
particularly useful for end to end tests where we don't have a comparison point
for the end result but we want to know when the result changes. Snapshot
testing is very common in numerical computing.

Usage example:

```
def test_foo(snapshot):
    K = 8000
    result = scipy.stats.binom.std(n=K, p=np.linspace(0.4, 0.6, 100)) / K
    snapshot(result, rtol=1e-7, atol=1e-7)
```

If you run `pytest --snapshot-update test_file.py::test_foo`, the snapshot will
be saved to disk. Then later when you run `pytest test_file.py::test_foo`, the
`snapshot(...)` call will automatically load that object and compare against it.

It's fine to call `snapshot(...)` multiple times in a test. The snapshot
filename will have an incremented counter indicating which call it is.

When debugging a snapshot test, you can use `snapshot.get(...)` to get the
value of the snapshot. If you are using the `TextSerializer`, you can also
look at the snapshot file directly. Pandas DataFrame objects are saved as csv
"""
import os
import pickle
from pathlib import Path

import jax.numpy
import numpy as np
import pandas as pd
import pytest


def pytest_addoption(parser):
    """
    Exposes snapshot plugin configuration to pytest.
    https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_addoption
    """
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        dest="update_snapshots",
        help="Update snapshots",
    )


def check_exists(snapshot_path):
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"Snapshot file not found: {snapshot_path}."
            " Did you forget to run with --snapshot-update?"
        )


class Pickler:
    @staticmethod
    def serialize(filebase, obj):
        with open(filebase + ".pkl", "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def deserialize(filebase, obj):
        filename = filebase + ".pkl"
        check_exists(filename)
        with open(filename, "rb") as f:
            return pickle.load(f)


class TextSerializer:
    @staticmethod
    def serialize(filebase, obj):
        if isinstance(obj, pd.DataFrame):
            filename = filebase + ".csv"
            # in all our dataframes, the index is meaningless, so we do not
            # save it here.
            obj.to_csv(filename, index=False)
        elif isinstance(obj, np.ndarray) or isinstance(obj, jax.numpy.DeviceArray):
            filename = filebase + ".txt"
            np.savetxt(filename, obj)
        else:
            raise ValueError(
                f"TextSerializer cannot serialize {type(obj)}."
                " Try calling snapshot(obj, serializer=Pickler)."
            )

    @staticmethod
    def deserialize(filebase, obj):
        if isinstance(obj, pd.DataFrame):
            filename = filebase + ".csv"
            check_exists(filename)
            return pd.read_csv(filename)
        elif isinstance(obj, np.ndarray) or isinstance(obj, jax.numpy.DeviceArray):
            filename = filebase + ".txt"
            check_exists(filename)
            return np.loadtxt(filename)
        else:
            raise ValueError(
                f"TextSerializer cannot deserialize {type(obj)}."
                " Try calling snapshot(obj, serializer=Pickler)."
            )


def pd_np_compare(actual, expected, **kwargs):
    if isinstance(actual, pd.DataFrame):
        # check_dtype=False is needed when we're using TextSerializer because
        # uint64 will be reloaded as int64
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, **kwargs)
    elif isinstance(actual, np.ndarray) or isinstance(actual, jax.numpy.DeviceArray):
        np.testing.assert_allclose(actual, expected, **kwargs)
    else:
        assert actual == expected


class SnapshotAssertion:
    def __init__(
        self,
        *,
        update_snapshots,
        request,
        default_comparator=pd_np_compare,
        default_serializer=TextSerializer,
    ):
        self.update_snapshots = update_snapshots
        self.request = request
        self.default_comparator = default_comparator
        self.default_serializer = default_serializer
        self.calls = 0

    def _get_filebase(self):
        test_folder = Path(self.request.fspath).parent
        test_name = self.request.node.name
        return test_folder.joinpath("__snapshot__", test_name + f"_{self.calls}")

    def get(self, obj, serializer=None):
        """
        This is a debugging helper function to see the values of the snapshot.

        Args:
            obj: The object to compare against. This is needed here to
                 determine the file extension.
            serializer: The serializer for loading the snapshot. Defaults to
                None which means we will use default_serializer. Unless
                default_serializer has been changed, this is TextSerializer, which
                will save the object as a .txt or .csv depending on whether it's a
                pd.DataFrame or np.ndarray.

        Returns:
            The snapshotted object.
        """
        if serializer is None:
            serializer = self.default_serializer

        return serializer.deserialize(str(self._get_filebase()), obj)

    def __call__(self, obj, comparator=None, serializer=None, **comparator_kwargs):
        if comparator is None:
            comparator = self.default_comparator
        if serializer is None:
            serializer = self.default_serializer

        # We provide the serializer with a filename without an extension. The
        # serializer can choose what extension to use.
        filebase = self._get_filebase()
        if self.update_snapshots:
            filebase.parent.mkdir(exist_ok=True)
            serializer.serialize(str(filebase), obj)
        else:
            comparator(
                obj, serializer.deserialize(str(filebase), obj), **comparator_kwargs
            )
        self.calls += 1


@pytest.fixture
def snapshot(request):
    return SnapshotAssertion(
        update_snapshots=request.config.option.update_snapshots,
        request=request,
    )
