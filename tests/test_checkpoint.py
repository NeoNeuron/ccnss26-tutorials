import numpy as np
import pytest
from pathlib import Path
from ccnss_helpers._checkpoint import save_checkpoint, load_or_compute


def test_save_and_load_roundtrip(tmp_path):
    save_checkpoint("foo", root=tmp_path, x=np.arange(5), y=np.eye(3))
    loaded = load_or_compute("foo", root=tmp_path, fn=lambda: pytest.fail("should not recompute"))
    assert np.array_equal(loaded["x"], np.arange(5))
    assert np.array_equal(loaded["y"], np.eye(3))


def test_recomputes_if_missing(tmp_path):
    called = []

    def fn():
        called.append(1)
        return {"a": np.array([1, 2, 3])}

    result = load_or_compute("bar", root=tmp_path, fn=fn)
    assert called == [1]
    assert np.array_equal(result["a"], np.array([1, 2, 3]))
    # Second call must hit cache.
    result2 = load_or_compute("bar", root=tmp_path, fn=lambda: pytest.fail("cache miss"))
    assert np.array_equal(result2["a"], np.array([1, 2, 3]))


def test_returns_arrays_not_npzfile(tmp_path):
    """load_or_compute must return a dict of arrays, not a lazy NpzFile."""
    save_checkpoint("baz", root=tmp_path, x=np.arange(5))
    loaded = load_or_compute("baz", root=tmp_path, fn=lambda: pytest.fail())
    assert isinstance(loaded, dict)
    assert isinstance(loaded["x"], np.ndarray)
