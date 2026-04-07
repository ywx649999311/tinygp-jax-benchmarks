"""Benchmark constants and profile definitions."""

from __future__ import annotations

from dataclasses import dataclass

SIGMA = 1.5
RHO = 2.5
JITTER = 0.1
SEED = 49_382
TINYGP_REF = "git+https://github.com/dfm/tinygp.git@main"
SUPPORTED_PROFILES = ("smoke", "ci", "full")

FULL_SCENARIO_SIZES = {
    "quasisep_cpu": (10, 20, 100, 200, 1_000, 2_000, 10_000),
    "quasisep_matern32_cpu": (
        10,
        20,
        100,
        200,
        1_000,
        2_000,
        10_000,
        # 20_000,
        # 100_000,
    ),
    "quasisep_matern52_cpu": (
        10,
        20,
        100,
        200,
        1_000,
        2_000,
        10_000,
        # 20_000,
        # 100_000,
    ),
}

SMOKE_SCENARIO_SIZES = {
    "quasisep_cpu": (10, 50, 200, 500, 2000),
    "quasisep_matern32_cpu": (10, 50, 200, 500, 2000),
    "quasisep_matern52_cpu": (10, 50, 200, 500, 2000),
}


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    samples: int
    scenario_sizes: dict[str, tuple[int, ...]]


PROFILE_CONFIGS = {
    "smoke": ProfileConfig(
        name="smoke",
        samples=1000,
        scenario_sizes=SMOKE_SCENARIO_SIZES,
    ),
    "ci": ProfileConfig(
        name="ci",
        samples=100,
        scenario_sizes=FULL_SCENARIO_SIZES,
    ),
    "full": ProfileConfig(
        name="full",
        samples=100,
        scenario_sizes=FULL_SCENARIO_SIZES,
    ),
}
