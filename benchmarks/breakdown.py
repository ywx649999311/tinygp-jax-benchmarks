"""CLI for benchmarking the gp.log_probability call tree."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import jax

from benchmarks.config import PROFILE_CONFIGS, SUPPORTED_PROFILES, TINYGP_REF
from benchmarks.data import make_dataset
from benchmarks.scenarios import detect_platform, get_scenario_specs, prepare_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=SUPPORTED_PROFILES,
        help="Benchmark profile to run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the JSON breakdown output.",
    )
    return parser


def _block(value):
    return jax.block_until_ready(value)


def _compile_callable(func, *args):
    compiled_fn = jax.jit(func).lower(*args).compile()
    _block(compiled_fn(*args))
    return compiled_fn


def _summarize_samples(
    scenario: str,
    size: int,
    stage: str,
    samples: list[float],
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "n": size,
        "stage": stage,
        "samples": len(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _measure_callable(compiled_fn, *args, sample_count: int) -> list[float]:
    durations = []
    for _ in range(sample_count):
        started = time.perf_counter()
        _block(compiled_fn(*args))
        durations.append(time.perf_counter() - started)
    return durations


def run_profile(profile_name: str) -> dict[str, object]:
    profile = PROFILE_CONFIGS[profile_name]
    results = []
    scenarios = get_scenario_specs()

    for scenario_name, sizes in profile.scenario_sizes.items():
        scenario = scenarios[scenario_name]
        for size in sizes:
            x, y = make_dataset(size)
            prepared_x, prepared_y = prepare_inputs(x, y)

            compiled_build_gp = _compile_callable(scenario.build_gp, prepared_x)
            build_gp_samples = _measure_callable(
                compiled_build_gp,
                prepared_x,
                sample_count=profile.samples,
            )
            results.append(
                _summarize_samples(
                    scenario=scenario_name,
                    size=size,
                    stage="build_gp",
                    samples=build_gp_samples,
                )
            )

            gp = _block(compiled_build_gp(prepared_x))
            centered_y = prepared_y - gp.loc
            alpha = _block(gp._get_alpha(prepared_y))
            kernel = gp.kernel
            qsm = gp.solver.matrix

            stage_functions = {
                "kernel.to_symm_qsm": (
                    lambda k, x: k.to_symm_qsm(x),
                    (kernel, prepared_x),
                ),
                "matrix.cholesky": (
                    lambda m: m.cholesky(),
                    (qsm,),
                ),
                "log_probability": (
                    lambda gp, y: gp.log_probability(y),
                    (gp, prepared_y),
                ),
                "_get_alpha": (
                    lambda gp, y: gp._get_alpha(y),
                    (gp, prepared_y),
                ),
                "solver.solve_triangular": (
                    lambda gp, y: gp.solver.solve_triangular(y),
                    (gp, centered_y),
                ),
                "factor.solve": (
                    lambda gp, y: gp.solver.factor.solve(y),
                    (gp, centered_y),
                ),
                "_compute_log_prob": (
                    lambda gp, a: gp._compute_log_prob(a),
                    (gp, alpha),
                ),
                "solver.normalization": (
                    lambda gp: gp.solver.normalization(),
                    (gp,),
                ),
            }

            for stage_name, (stage_fn, stage_args) in stage_functions.items():
                compiled_fn = _compile_callable(stage_fn, *stage_args)
                samples = _measure_callable(
                    compiled_fn,
                    *stage_args,
                    sample_count=profile.samples,
                )
                results.append(
                    _summarize_samples(
                        scenario=scenario_name,
                        size=size,
                        stage=stage_name,
                        samples=samples,
                    )
                )

    return {
        "python": platform.python_version(),
        "jax": sys.modules["jax"].__version__,
        "tinygp_ref": TINYGP_REF,
        "platform": detect_platform(),
        "profile": profile_name,
        "results": results,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = run_profile(args.profile)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
