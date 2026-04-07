"""Benchmark scenarios adapted from the tinygp benchmarks notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from benchmarks.config import JITTER, RHO, SIGMA
from benchmarks.env import configure_cpu_environment

configure_cpu_environment()

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import tinygp

ScenarioFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
CPU_DEVICE = jax.local_devices(backend="cpu")[0]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    build_gp: Callable[[jnp.ndarray], tinygp.GaussianProcess]
    loss_fn: ScenarioFn
    jit_loss_fn: ScenarioFn


def _build_quasisep_exp_gp(x: jnp.ndarray) -> tinygp.GaussianProcess:
    kernel = tinygp.kernels.quasisep.Exp(sigma=SIGMA, scale=RHO)
    return tinygp.GaussianProcess(kernel, x, diag=JITTER**2, assume_sorted=True)

def _build_quasisep_matern32_gp(x: jnp.ndarray) -> tinygp.GaussianProcess:
    kernel = tinygp.kernels.quasisep.Matern32(sigma=SIGMA, scale=RHO)
    return tinygp.GaussianProcess(kernel, x, diag=JITTER**2, assume_sorted=True)


def _build_quasisep_matern52_gp(x: jnp.ndarray) -> tinygp.GaussianProcess:
    kernel = tinygp.kernels.quasisep.Matern52(sigma=SIGMA, scale=RHO)
    return tinygp.GaussianProcess(kernel, x, diag=JITTER**2, assume_sorted=True)


def _make_loss_fn(build_gp: Callable[[jnp.ndarray], tinygp.GaussianProcess]) -> ScenarioFn:
    def loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        gp = build_gp(x)
        return -gp.log_probability(y)

    return loss


def _make_scenario_spec(
    name: str,
    build_gp: Callable[[jnp.ndarray], tinygp.GaussianProcess],
) -> ScenarioSpec:
    loss_fn = _make_loss_fn(build_gp)
    return ScenarioSpec(
        name=name,
        build_gp=build_gp,
        loss_fn=loss_fn,
        jit_loss_fn=jax.jit(loss_fn),
    )


SCENARIOS = {
    "quasisep_cpu": _make_scenario_spec("quasisep_cpu", _build_quasisep_exp_gp),
    "quasisep_matern32_cpu": _make_scenario_spec(
        "quasisep_matern32_cpu",
        _build_quasisep_matern32_gp,
    ),
    "quasisep_matern52_cpu": _make_scenario_spec(
        "quasisep_matern52_cpu",
        _build_quasisep_matern52_gp,
    ),
}


def prepare_inputs(x: np.ndarray, y: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (
        jax.device_put(jnp.asarray(x), CPU_DEVICE),
        jax.device_put(jnp.asarray(y), CPU_DEVICE),
    )


def get_scenarios() -> dict[str, ScenarioFn]:
    return {name: spec.jit_loss_fn for name, spec in SCENARIOS.items()}


def get_scenario_specs() -> dict[str, ScenarioSpec]:
    return dict(SCENARIOS)


def evaluate_scenario(name: str, x: np.ndarray, y: np.ndarray) -> float:
    scenarios = get_scenario_specs()
    if name not in scenarios:
        raise KeyError(f"unknown scenario: {name}")
    prepared_x, prepared_y = prepare_inputs(x, y)
    value = scenarios[name].jit_loss_fn(prepared_x, prepared_y).block_until_ready()
    return float(value)


def detect_platform() -> str:
    return CPU_DEVICE.platform
