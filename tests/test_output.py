from __future__ import annotations

import json
from pathlib import Path

from benchmarks.compare import render_markdown, save_plot
from benchmarks.run import run_profile


def test_run_profile_output_schema() -> None:
    payload = run_profile("smoke")

    assert payload["profile"] == "smoke"
    assert payload["platform"] == "cpu"
    assert payload["results"]

    first = payload["results"][0]
    assert set(first) == {
        "scenario",
        "n",
        "samples",
        "median_s",
        "mean_s",
        "stdev_s",
    }

    encoded = json.dumps(payload)
    assert '"tinygp_ref"' in encoded


def test_render_markdown_includes_ratio_table() -> None:
    baseline = {
        "jax": "0.4.31",
        "platform": "cpu",
        "profile": "ci",
        "results": [
            {
                "scenario": "exact_cpu",
                "n": 10,
                "samples": 3,
                "median_s": 0.11,
                "mean_s": 0.10,
                "stdev_s": 0.01,
            }
        ],
    }
    candidate = {
        "jax": "0.9.1",
        "platform": "cpu",
        "profile": "ci",
        "results": [
            {
                "scenario": "exact_cpu",
                "n": 10,
                "samples": 3,
                "median_s": 0.21,
                "mean_s": 0.20,
                "stdev_s": 0.01,
            }
        ],
    }

    markdown = render_markdown(baseline, candidate)

    assert "| scenario | n | baseline mean (s) | candidate mean (s) | ratio |" in markdown
    assert "| exact_cpu | 10 | 0.100000 | 0.200000 | 2.000 |" in markdown


def test_save_plot_writes_plot_file(tmp_path: Path) -> None:
    baseline = {
        "jax": "0.4.31",
        "platform": "cpu",
        "profile": "ci",
        "results": [
            {"scenario": "exact_cpu", "n": 10, "samples": 3, "median_s": 0.11, "mean_s": 0.10, "stdev_s": 0.01},
            {"scenario": "exact_cpu", "n": 100, "samples": 3, "median_s": 0.20, "mean_s": 0.20, "stdev_s": 0.01},
        ],
    }
    candidate = {
        "jax": "0.9.1",
        "platform": "cpu",
        "profile": "ci",
        "results": [
            {"scenario": "exact_cpu", "n": 10, "samples": 3, "median_s": 0.09, "mean_s": 0.08, "stdev_s": 0.01},
            {"scenario": "exact_cpu", "n": 100, "samples": 3, "median_s": 0.16, "mean_s": 0.15, "stdev_s": 0.01},
        ],
    }

    output_path = tmp_path / "comparison.png"
    saved_path = save_plot(baseline, candidate, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
