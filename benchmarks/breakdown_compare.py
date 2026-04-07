"""CLI for comparing stage-by-stage benchmark breakdowns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline breakdown JSON path.")
    parser.add_argument("candidate", help="Candidate breakdown JSON path.")
    return parser


def _load_payload(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _row_key(row: dict[str, object]) -> tuple[str, int, str]:
    return str(row["scenario"]), int(row["n"]), str(row["stage"])


def render_markdown(baseline: dict[str, object], candidate: dict[str, object]) -> str:
    baseline_rows = {_row_key(row): row for row in baseline["results"]}
    candidate_rows = {_row_key(row): row for row in candidate["results"]}
    shared_keys = sorted(set(baseline_rows) & set(candidate_rows))

    lines = [
        "## tinygp benchmark breakdown comparison",
        "",
        f"- baseline JAX: `{baseline['jax']}`",
        f"- candidate JAX: `{candidate['jax']}`",
        f"- profile: `{candidate['profile']}`",
        "",
        "| scenario | n | stage | baseline mean (s) | candidate mean (s) | ratio |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]

    for key in shared_keys:
        baseline_row = baseline_rows[key]
        candidate_row = candidate_rows[key]
        baseline_mean = float(baseline_row["mean_s"])
        candidate_mean = float(candidate_row["mean_s"])
        ratio = candidate_mean / baseline_mean if baseline_mean else float("inf")
        lines.append(
            "| "
            f"{key[0]} | {key[1]} | {key[2]} | {baseline_mean:.6f} | {candidate_mean:.6f} | {ratio:.3f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    baseline = _load_payload(args.baseline)
    candidate = _load_payload(args.candidate)
    print(render_markdown(baseline, candidate), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
