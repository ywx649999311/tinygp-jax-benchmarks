"""CLI for running tinygp CPU benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from benchmarks.config import PROFILE_CONFIGS, SUPPORTED_PROFILES, TINYGP_REF
from benchmarks.data import make_dataset
from benchmarks.scenarios import detect_platform, get_scenarios, prepare_inputs


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
        help="Path to the JSON benchmark output.",
    )
    return parser


def _compile_scenario(scenario_fn, x, y):
    compiled_fn = scenario_fn.lower(x, y).compile()
    compiled_fn(x, y).block_until_ready()
    return compiled_fn


def _measure_samples(compiled_fn, x, y, sample_count: int) -> list[float]:
    durations = []
    for _ in range(sample_count):
        started = time.perf_counter()
        compiled_fn(x, y).block_until_ready()
        durations.append(time.perf_counter() - started)
    return durations


def run_profile(profile_name: str) -> dict[str, object]:
    profile = PROFILE_CONFIGS[profile_name]
    results = []
    scenarios = get_scenarios()

    for scenario_name, sizes in profile.scenario_sizes.items():
        scenario_fn = scenarios[scenario_name]
        for size in sizes:
            x, y = make_dataset(size)
            prepared_x, prepared_y = prepare_inputs(x, y)
            compiled_fn = _compile_scenario(
                scenario_fn=scenario_fn,
                x=prepared_x,
                y=prepared_y,
            )
            samples = _measure_samples(
                compiled_fn=compiled_fn,
                x=prepared_x,
                y=prepared_y,
                sample_count=profile.samples,
            )
            results.append(
                {
                    "scenario": scenario_name,
                    "n": size,
                    "samples": len(samples),
                    "median_s": statistics.median(samples),
                    "mean_s": statistics.fmean(samples),
                    "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                }
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
