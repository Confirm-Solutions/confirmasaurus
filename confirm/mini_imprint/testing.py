import os
import pickle
from pathlib import Path

import jax.numpy
import numpy as np
import pandas as pd
import pytest


def pytest_addoption(parser) -> None:
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


class Pickler:
    @staticmethod
    def serialize(filebase, obj):
        with open(filebase + ".pkl", "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def deserialize(filebase, obj):
        filename = filebase + ".pkl"
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Snapshot file not found: {filename}."
                " Did you forget to run with --snapshot-update?"
            )
        with open(filename, "rb") as f:
            return pickle.load(f)


class TextSerializer:
    @staticmethod
    def serialize(filebase, obj):
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(filebase + ".csv")
        elif isinstance(obj, np.ndarray) or isinstance(obj, jax.numpy.DeviceArray):
            np.savetxt(filebase + ".txt", obj)
        else:
            raise ValueError(
                f"TextSerializer cannot serialize {type(obj)}."
                " Try calling snapshot(obj, serializer=Pickler)."
            )

    @staticmethod
    def deserialize(filebase, obj):
        if isinstance(obj, pd.DataFrame):
            return pd.read_csv(filebase + ".csv")
        elif isinstance(obj, np.ndarray) or isinstance(obj, jax.numpy.DeviceArray):
            return np.loadtxt(filebase + ".txt")
        else:
            raise ValueError(
                f"TextSerializer cannot deserialize {type(obj)}."
                " Try calling snapshot(obj, serializer=Pickler)."
            )


def pd_np_compare(actual, expected, **kwargs):
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, **kwargs)
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
    """
    _summary_

    Args:
        request: _description_

    Returns:
        _description_
    """
    return SnapshotAssertion(
        update_snapshots=request.config.option.update_snapshots,
        request=request,
    )
