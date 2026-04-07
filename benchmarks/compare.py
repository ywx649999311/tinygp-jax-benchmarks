"""CLI for comparing two tinygp benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline JSON path, typically JAX 0.4.31.")
    parser.add_argument("candidate", help="Candidate JSON path, typically JAX 0.9.1.")
    parser.add_argument(
        "--plot-output",
        help="Optional path where a benchmark comparison plot should be written.",
    )
    return parser


def _load_payload(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _row_key(row: dict[str, object]) -> tuple[str, int]:
    return str(row["scenario"]), int(row["n"])


def render_markdown(baseline: dict[str, object], candidate: dict[str, object]) -> str:
    baseline_rows = {_row_key(row): row for row in baseline["results"]}
    candidate_rows = {_row_key(row): row for row in candidate["results"]}
    shared_keys = sorted(set(baseline_rows) & set(candidate_rows))

    lines = [
        "## tinygp CPU benchmark comparison",
        "",
        f"- baseline JAX: `{baseline['jax']}`",
        f"- candidate JAX: `{candidate['jax']}`",
        f"- profile: `{candidate['profile']}`",
        f"- platform: `{candidate['platform']}`",
        "",
        "| scenario | n | baseline mean (s) | candidate mean (s) | ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for key in shared_keys:
        baseline_row = baseline_rows[key]
        candidate_row = candidate_rows[key]
        baseline_mean = float(baseline_row["mean_s"])
        candidate_mean = float(candidate_row["mean_s"])
        ratio = candidate_mean / baseline_mean if baseline_mean else float("inf")
        lines.append(
            "| "
            f"{key[0]} | {key[1]} | {baseline_mean:.6f} | {candidate_mean:.6f} | {ratio:.3f} |"
        )

    return "\n".join(lines) + "\n"


def save_plot(
    baseline: dict[str, object],
    candidate: dict[str, object],
    output_path: str | Path,
) -> Path:
    baseline_rows = {}
    for row in baseline["results"]:
        baseline_rows.setdefault(str(row["scenario"]), []).append(row)

    candidate_rows = {}
    for row in candidate["results"]:
        candidate_rows.setdefault(str(row["scenario"]), []).append(row)

    scenario_names = sorted(set(baseline_rows) | set(candidate_rows))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, scenario_name in enumerate(scenario_names):
        color = color_cycle[index % len(color_cycle)]

        if scenario_name in baseline_rows:
            rows = sorted(baseline_rows[scenario_name], key=lambda row: int(row["n"]))
            ax.loglog(
                [int(row["n"]) for row in rows],
                [float(row["mean_s"]) for row in rows],
                "o-",
                color=color,
                lw=1.5,
                label=f"{scenario_name} ({baseline['jax']})",
            )

        if scenario_name in candidate_rows:
            rows = sorted(candidate_rows[scenario_name], key=lambda row: int(row["n"]))
            ax.loglog(
                [int(row["n"]) for row in rows],
                [float(row["mean_s"]) for row in rows],
                "s--",
                color=color,
                lw=1.5,
                label=f"{scenario_name} ({candidate['jax']})",
            )

    ax.set_xlabel("number of data points")
    ax.set_ylabel("runtime [s]")
    ax.set_title("tinygp CPU benchmark comparison")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    fig.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    baseline = _load_payload(args.baseline)
    candidate = _load_payload(args.candidate)
    print(render_markdown(baseline, candidate), end="")
    if args.plot_output:
        save_plot(baseline, candidate, args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
