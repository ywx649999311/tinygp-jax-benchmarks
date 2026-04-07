from __future__ import annotations

import numpy as np

from benchmarks.data import make_dataset


def test_make_dataset_is_deterministic() -> None:
    x1, y1 = make_dataset(8)
    x2, y2 = make_dataset(8)

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_make_dataset_rejects_non_positive_size() -> None:
    try:
        make_dataset(0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected make_dataset to reject non-positive sizes")
