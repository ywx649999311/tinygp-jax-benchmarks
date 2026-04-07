from __future__ import annotations

from benchmarks.breakdown_compare import render_markdown
from benchmarks.scenarios import get_scenario_specs


def test_get_scenario_specs_matches_current_quasisep_set() -> None:
    specs = get_scenario_specs()

    assert set(specs) == {
        "quasisep_cpu",
        "quasisep_matern32_cpu",
        "quasisep_matern52_cpu",
    }


def test_breakdown_render_markdown_includes_stage_ratios() -> None:
    baseline = {
        "jax": "0.4.31",
        "profile": "smoke",
        "results": [
            {
                "scenario": "quasisep_cpu",
                "n": 50,
                "stage": "jit_compile",
                "samples": 1,
                "median_s": 1.1,
                "mean_s": 1.1,
                "stdev_s": 0.0,
            }
        ],
    }
    candidate = {
        "jax": "0.9.1",
        "profile": "smoke",
        "results": [
            {
                "scenario": "quasisep_cpu",
                "n": 50,
                "stage": "jit_compile",
                "samples": 1,
                "median_s": 2.2,
                "mean_s": 2.2,
                "stdev_s": 0.0,
            }
        ],
    }

    markdown = render_markdown(baseline, candidate)

    assert "| scenario | n | stage | baseline mean (s) | candidate mean (s) | ratio |" in markdown
    assert "| quasisep_cpu | 50 | jit_compile | 1.100000 | 2.200000 | 2.000 |" in markdown
