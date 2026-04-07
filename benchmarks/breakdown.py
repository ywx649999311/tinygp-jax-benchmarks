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


def _measure_callable(func, sample_count: int) -> list[float]:
    _block(func())
    durations = []
    for _ in range(sample_count):
        started = time.perf_counter()
        _block(func())
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

            gp = scenario.build_gp(prepared_x)
            centered_y = prepared_y - gp.loc
            alpha = _block(gp._get_alpha(prepared_y))

            stage_functions = {
                "log_probability": lambda gp=gp, y=prepared_y: gp.log_probability(y),
                "_get_alpha": lambda gp=gp, y=prepared_y: gp._get_alpha(y),
                "solver.solve_triangular": (
                    lambda gp=gp, y=centered_y: gp.solver.solve_triangular(y)
                ),
                "factor.solve": lambda gp=gp, y=centered_y: gp.solver.factor.solve(y),
                "_compute_log_prob": lambda gp=gp, a=alpha: gp._compute_log_prob(a),
                "solver.normalization": lambda gp=gp: gp.solver.normalization(),
            }

            for stage_name, stage_fn in stage_functions.items():
                samples = _measure_callable(stage_fn, profile.samples)
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
