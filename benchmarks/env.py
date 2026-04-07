"""Runtime environment helpers for CPU-only JAX benchmarking."""

from __future__ import annotations

import os

_XLA_SINGLE_THREAD_FLAG = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)


def configure_cpu_environment() -> None:
    """Force a reproducible CPU-only JAX execution environment."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if _XLA_SINGLE_THREAD_FLAG not in xla_flags:
        combined = f"{xla_flags} {_XLA_SINGLE_THREAD_FLAG}".strip()
        os.environ["XLA_FLAGS"] = combined
