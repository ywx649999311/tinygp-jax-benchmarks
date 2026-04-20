"""Time matrix.cholesky for Matern52 in float32 vs float64.

Bypasses scenarios.py so we control the dtype directly without the
jax_enable_x64 flag that the main benchmark harness forces on.

Usage:
    uv run python -m benchmarks.dtype_check --output results/dtype-check.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.config import RHO, SIGMA
from benchmarks.env import configure_cpu_environment

# Must run before any JAX CPU kernels are initialised.
configure_cpu_environment()

from tinygp.kernels.quasisep import Matern52  # noqa: E402

SIZES = [10, 50, 200, 500, 2000]
SAMPLES = 200


def _build_qsm(n: int, dtype):
    x = jnp.linspace(0.0, 10.0, n, dtype=dtype)
    sigma = jnp.array(SIGMA, dtype=dtype)
    scale = jnp.array(RHO, dtype=dtype)
    kernel = Matern52(sigma=sigma, scale=scale)
    qsm = kernel.to_symm_qsm(x)
    # Matern52's transition-matrix constants are Python floats (float64 by
    # default), so with jax_enable_x64=True the QSM arrays can silently
    # promote to float64 even when x was float32.  Cast everything back.
    return jax.tree.map(lambda a: a.astype(dtype), qsm)


def _time_cholesky(qsm, samples: int) -> list[float]:
    compiled = jax.jit(lambda m: m.cholesky()).lower(qsm).compile()
    jax.block_until_ready(compiled(qsm))  # warmup
    durations = []
    for _ in range(samples):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(qsm))
        durations.append(time.perf_counter() - t0)
    return durations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Path to JSON output.")
    return parser


def main() -> int:
    # float64 must be enabled before timing — this script tests both,
    # so we enable it globally and cast explicitly per dtype.
    jax.config.update("jax_enable_x64", True)

    parser = build_parser()
    args = parser.parse_args()

    results = []
    for dtype_name, dtype in [("float32", jnp.float32), ("float64", jnp.float64)]:
        for n in SIZES:
            print(f"  {dtype_name}  n={n}...", end=" ", flush=True)
            qsm = _build_qsm(n, dtype)
            samples = _time_cholesky(qsm, SAMPLES)
            results.append({
                "dtype": dtype_name,
                "n": n,
                "samples": len(samples),
                "mean_s": statistics.fmean(samples),
                "median_s": statistics.median(samples),
                "stdev_s": statistics.stdev(samples),
            })
            print(f"{statistics.median(samples)*1e6:.1f} µs")

    payload = {
        "jax": jax.__version__,
        "python": platform.python_version(),
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
