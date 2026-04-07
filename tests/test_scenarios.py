from __future__ import annotations

import math

from benchmarks.data import make_dataset
from benchmarks.scenarios import detect_platform, evaluate_scenario, get_scenarios


def test_scenarios_return_finite_scalars() -> None:
    x, y = make_dataset(10)

    for scenario_name in get_scenarios():
        value = evaluate_scenario(scenario_name, x, y)
        assert isinstance(value, float)
        assert math.isfinite(value)


def test_platform_detection_reports_cpu() -> None:
    assert detect_platform() == "cpu"
