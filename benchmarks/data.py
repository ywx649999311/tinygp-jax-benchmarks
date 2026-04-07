"""Deterministic benchmark inputs."""

from __future__ import annotations

import numpy as np

from benchmarks.config import JITTER, SEED


def make_dataset(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate the notebook-derived deterministic benchmark dataset."""
    if size <= 0:
        raise ValueError("size must be positive")

    random = np.random.default_rng(SEED)
    x = np.sort(random.uniform(0, 10, size))
    y = np.sin(x) + JITTER * random.normal(0, 1, len(x))
    return x, y
