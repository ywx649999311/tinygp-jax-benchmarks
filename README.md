# tinygp JAX Benchmarks

This repository benchmarks CPU-only `tinygp` performance against two pinned JAX environments:

- `jax==0.4.31`
- `jax==0.9.1`

The benchmark logic is adapted from `dfm/tinygp`'s `docs/benchmarks.ipynb`, but it now benchmarks only the CPU quasisep `tinygp` paths:

- `quasisep_cpu`: quasisep `tinygp.GaussianProcess` likelihood on CPU using `Exp`
- `quasisep_matern32_cpu`: quasisep `tinygp.GaussianProcess` likelihood on CPU using `Matern32`
- `quasisep_matern52_cpu`: quasisep `tinygp.GaussianProcess` likelihood on CPU using `Matern52`

The harness forces CPU execution and uses the same deterministic notebook constants:

- `sigma = 1.5`
- `rho = 2.5`
- `jitter = 0.1`
- RNG seed `49382`

Each scenario builds a `tinygp.GaussianProcess`, wraps `GP.log_probability(y)` in a loss function, lowers and compiles that JITed loss once per input size, and benchmarks only the compiled callable.

## Local Usage

Install one JAX lane plus the test tools in a single `uv` resolve:

```bash
uv sync --extra jax0431 --group test
```

or:

```bash
uv sync --extra jax091 --group test
```

Run benchmarks:

```bash
uv run python -m benchmarks.run --profile smoke --output results/jax0431-smoke.json
uv run python -m benchmarks.run --profile ci --output results/jax091-ci.json
```

Compare two runs:

```bash
uv run python -m benchmarks.compare results/jax0431-ci.json results/jax091-ci.json
```

Save both the Markdown comparison and a notebook-style plot:

```bash
uv run python -m benchmarks.compare \
  results/jax0431-ci.json \
  results/jax091-ci.json \
  --plot-output results/comparison.png > results/summary.md
```

Run a `gp.log_probability` call-tree breakdown to see where the candidate slows down:

```bash
uv run python -m benchmarks.breakdown --profile smoke --output results/jax0431-breakdown.json
uv sync --extra jax091 --group test
uv run python -m benchmarks.breakdown --profile smoke --output results/jax091-breakdown.json
uv run python -m benchmarks.breakdown_compare \
  results/jax0431-breakdown.json \
  results/jax091-breakdown.json > results/breakdown-summary.md
```

The breakdown report benchmarks the internal call path behind `gp.log_probability` for each scenario and size:

- `log_probability`
- `_get_alpha`
- `solver.solve_triangular`
- `factor.solve`
- `_compute_log_prob`
- `solver.normalization`

## Profiles

- `smoke`: reduced size ladder with 20 timed samples per size
- `ci`: full notebook size ladder with 20 timed samples per size
- `full`: full notebook size ladder with 20 timed samples per size

## Output Schema

Each benchmark run produces JSON with:

- `python`
- `jax`
- `tinygp_ref`
- `platform`
- `profile`
- `results`

Each row in `results` contains:

- `scenario`
- `n`
- `samples`
- `median_s`
- `mean_s`
- `stdev_s`

## GitHub Actions

- `.github/workflows/benchmark.yml` runs the full two-version benchmark matrix manually via `workflow_dispatch`, uploads JSON artifacts, and appends a Markdown comparison table to the job summary.
- `.github/workflows/pr-smoke.yml` runs a lightweight smoke profile on pull requests using `jax==0.9.1`.
