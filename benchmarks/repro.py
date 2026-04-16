"""Standalone lax.scan timing reproducer — no tinygp dependency.

Prints one line with the median latency for the scan body that
mirrors the tinygp Matern52 cholesky loop (n=2000, m=3, float64).

Usage:
    uv run python -m benchmarks.repro
"""

import statistics
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

N, M, SAMPLES = 2000, 3, 200
dtype = jnp.float64

key = jax.random.PRNGKey(0)
d = jax.random.uniform(key, (N,), dtype=dtype, minval=1.0, maxval=2.0)
p = jax.random.normal(key, (N, M), dtype=dtype) * 0.1
q = jax.random.normal(key, (N, M), dtype=dtype) * 0.1
a = jax.random.normal(key, (N, M, M), dtype=dtype) * 0.1
init = jnp.zeros((M, M), dtype=dtype)


def body(carry, data):
    fp = carry
    dk, pk, qk, ak = data
    ck = jnp.sqrt(dk - pk @ fp @ pk)
    tmp = fp @ ak.T
    wk = (qk - pk @ tmp) / ck
    return ak @ tmp + jnp.outer(wk, wk), (ck, wk)


def main() -> int:
    scan_fn = lambda: jax.lax.scan(body, init, (d, p, q, a))
    compiled = jax.jit(scan_fn).lower().compile()
    jax.block_until_ready(compiled())  # warmup

    ts = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled())
        ts.append(time.perf_counter() - t0)

    print(
        f"JAX {jax.__version__}  n={N}  m={M}  dtype={dtype.dtype}  "
        f"median={statistics.median(ts) * 1e6:.1f} µs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
