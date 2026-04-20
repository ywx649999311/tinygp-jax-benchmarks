"""Standalone lax.scan timing reproducer — no tinygp dependency.

Prints one line with the median latency for the scan body that
mirrors the tinygp Matern52 cholesky loop.

Usage:
    uv run python -m benchmarks.repro
    uv run python -m benchmarks.repro --n 10000
    uv run python -m benchmarks.repro --output results/bisect/jax0431-repro.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

M, SAMPLES = 3, 200
dtype = jnp.float64


def body(carry, data):
    fp = carry
    dk, pk, qk, ak = data
    ck = jnp.sqrt(dk - pk @ fp @ pk)
    tmp = fp @ ak.T
    wk = (qk - pk @ tmp) / ck
    return ak @ tmp + jnp.outer(wk, wk), (ck, wk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000, metavar="N",
                        help="Number of scan iterations (default: 2000).")
    parser.add_argument("--output", metavar="FILE",
                        help="Write results as JSON to FILE (optional).")
    args = parser.parse_args()

    n = args.n
    key = jax.random.PRNGKey(0)
    d = jax.random.uniform(key, (n,), dtype=dtype, minval=1.0, maxval=2.0)
    p = jax.random.normal(key, (n, M), dtype=dtype) * 0.1
    q = jax.random.normal(key, (n, M), dtype=dtype) * 0.1
    a = jax.random.normal(key, (n, M, M), dtype=dtype) * 0.1
    init = jnp.zeros((M, M), dtype=dtype)

    scan_fn = lambda: jax.lax.scan(body, init, (d, p, q, a))
    compiled = jax.jit(scan_fn).lower().compile()
    jax.block_until_ready(compiled())  # warmup

    ts = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled())
        ts.append(time.perf_counter() - t0)

    median_s = statistics.median(ts)
    print(
        f"JAX {jax.__version__}  n={n}  m={M}  dtype={dtype.dtype}  "
        f"median={median_s * 1e6:.1f} µs"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "jax": jax.__version__,
            "n": n,
            "m": M,
            "dtype": str(dtype.dtype),
            "median_s": median_s,
            "samples": SAMPLES,
        }, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
