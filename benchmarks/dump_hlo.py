"""Dump StableHLO for quasisep scenarios and the minimal lax.scan reproducer.

Run once per JAX version, then diff:

    uv sync --extra jax0431
    uv run python -m benchmarks.dump_hlo --output results/hlo-jax0431

    uv sync --extra jax091
    uv run python -m benchmarks.dump_hlo --output results/hlo-jax091

    uv run python -m benchmarks.dump_hlo \\
        --diff results/hlo-jax0431/repro.txt results/hlo-jax091/repro.txt \\
        > results/hlo-repro-diff.txt

The --diff output is plain text with no ANSI codes — safe to paste into GitHub.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.data import make_dataset
from benchmarks.scenarios import get_scenario_specs, prepare_inputs

# ---------------------------------------------------------------------------
# Minimal lax.scan reproducer (no tinygp dependency)
# ---------------------------------------------------------------------------

_N, _M = 2000, 3
_dtype = jnp.float64


def _repro_body(carry, data):
    fp = carry
    dk, pk, qk, ak = data
    ck = jnp.sqrt(dk - pk @ fp @ pk)
    tmp = fp @ ak.T
    wk = (qk - pk @ tmp) / ck
    return ak @ tmp + jnp.outer(wk, wk), (ck, wk)


def _dump_repro() -> str:
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)
    d = jax.random.uniform(key, (_N,), dtype=_dtype, minval=1.0, maxval=2.0)
    p = jax.random.normal(key, (_N, _M), dtype=_dtype) * 0.1
    q = jax.random.normal(key, (_N, _M), dtype=_dtype) * 0.1
    a = jax.random.normal(key, (_N, _M, _M), dtype=_dtype) * 0.1
    init = jnp.zeros((_M, _M), dtype=_dtype)

    lowered = jax.jit(lambda: jax.lax.scan(_repro_body, init, (d, p, q, a))).lower()

    return "\n".join([
        f"scenario:  repro (standalone, no tinygp)",
        f"jax:       {jax.__version__}",
        f"n:         {_N}",
        f"m:         {_M}",
        f"dtype:     {_dtype}",
        "",
        "=" * 72,
        "PRE-OPTIMIZATION (StableHLO from .lower().as_text())",
        "=" * 72,
        "",
        lowered.as_text(),
    ])


# ---------------------------------------------------------------------------
# Tinygp scenario dump
# ---------------------------------------------------------------------------

def _dump_scenario(scenario_name: str, size: int) -> str:
    specs = get_scenario_specs()
    scenario = specs[scenario_name]

    x, y = make_dataset(size)
    px, _ = prepare_inputs(x, y)

    gp = jax.block_until_ready(scenario.build_gp(px))
    qsm = gp.solver.matrix

    lowered = jax.jit(lambda m: m.cholesky()).lower(qsm)

    return "\n".join([
        f"scenario:  {scenario_name}",
        f"jax:       {jax.__version__}",
        f"n:         {size}",
        "",
        "=" * 72,
        "PRE-OPTIMIZATION (StableHLO from .lower().as_text())",
        "=" * 72,
        "",
        lowered.as_text(),
    ])


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _diff(baseline: Path, candidate: Path) -> None:
    diff = list(
        difflib.unified_diff(
            baseline.read_text().splitlines(keepends=True),
            candidate.read_text().splitlines(keepends=True),
            fromfile=str(baseline),
            tofile=str(candidate),
            n=3,
        )
    )
    if not diff:
        print("No differences found.")
    else:
        print(f"{len(diff)} diff lines across {sum(1 for l in diff if l.startswith('@@'))} chunks\n")
        sys.stdout.writelines(diff)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output",
        metavar="DIR",
        help="Directory to write HLO text files into.",
    )
    group.add_argument(
        "--diff",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="Print a plain-text unified diff of two HLO dump files.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Input size n for tinygp scenario dumps (ignored for 'repro', default: 500).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["quasisep_cpu", "quasisep_matern32_cpu", "quasisep_matern52_cpu"],
        help="Scenarios to dump. Use 'repro' for the standalone reproducer (default: all three quasisep scenarios).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.diff:
        _diff(Path(args.diff[0]), Path(args.diff[1]))
        return 0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name in args.scenarios:
        print(f"  {scenario_name}...", end=" ", flush=True)
        if scenario_name == "repro":
            text = _dump_repro()
        else:
            text = _dump_scenario(scenario_name, args.size)
        out_path = out_dir / f"{scenario_name}.txt"
        out_path.write_text(text)
        print(f"written to {out_path}")

    print(f"\nJAX {jax.__version__} HLO saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
