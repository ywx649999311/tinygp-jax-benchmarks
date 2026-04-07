# `gp.log_probability` Call Graph

This document expands the quasisep benchmark path down to the lowest practical
level in the installed `tinygp` source used by this repository. It covers both:

- the `build_gp` stage now benchmarked in `benchmarks.breakdown`
- the runtime path for `gp.log_probability(y)` and its internal sub-stages

The graph below reflects the current local `tinygp` implementation in
`.venv/lib/python3.11/site-packages/tinygp/`.

## Mermaid

```mermaid
flowchart TD
    A[build_gp(x)] --> A1[_build_quasisep_exp_gp / _build_quasisep_matern32_gp / _build_quasisep_matern52_gp]
    A1 --> A2[tinygp.GaussianProcess.__init__]

    A2 --> A3[jax.vmap(mean_function)(X)]
    A2 --> A4[Diagonal(diag=broadcast(jitter^2))]
    A2 --> A5[QuasisepSolver.__init__]

    A5 --> A6[kernel.to_symm_qsm(X)]
    A5 --> A7[noise.to_qsm]
    A5 --> A8[SymmQSM.__add__]
    A8 --> A9[ops.elementwise_add]
    A9 --> A10[deconstruct]
    A9 --> A11[add_two]
    A9 --> A12[construct]
    A12 --> A13[matrix]
    A13 --> A14[SymmQSM.cholesky]
    A14 --> A15[jax.lax.scan over impl]
    A15 --> A16[LowerTriQSM factor]

    A6 --> B1[stationary_covariance]
    A6 --> B2[vmap transition_matrix]
    A6 --> B3[vmap observation_model]
    B1 --> B4[Pinf]
    B2 --> B5[a]
    B3 --> B6[h]
    B6 --> B7[q = h]
    B4 --> B8[p = h @ Pinf]
    B8 --> B9[d = sum(p * q)]
    B8 --> B10[vmap x @ y with a]
    B10 --> B11[StrictLowerTriQSM(p, q, a)]
    B9 --> B12[DiagQSM(d)]
    B11 --> B13[SymmQSM(diag, lower)]

    B7 --> C1[Kernel-specific leaves]
    C1 --> C2[Exp: observation_model -> [sigma]]
    C1 --> C3[Exp: transition_matrix -> exp(-dt / scale)]
    C1 --> C4[Matern32: observation_model -> [sigma, 0]]
    C1 --> C5[Matern32: transition_matrix -> exp(-f dt) * 2x2 polynomial]
    C1 --> C6[Matern52: observation_model -> [sigma, 0, 0]]
    C1 --> C7[Matern52: transition_matrix -> exp(-f dt) * 3x3 polynomial]

    D[gp.log_probability(y)] --> D1[gp._get_alpha(y)]
    D --> D2[gp._compute_log_prob(alpha)]

    D1 --> D3[y - gp.loc]
    D3 --> D4[solver.solve_triangular]
    D4 --> D5[factor.solve]
    D5 --> D6[LowerTriQSM.solve]
    D6 --> D7[jax.lax.scan forward substitution]
    D7 --> D8[alpha]

    D2 --> D9[-0.5 * sum(alpha^2)]
    D2 --> D10[solver.normalization]
    D10 --> D11[sum(log(factor.diag.d))]
    D10 --> D12[0.5 * N * log(2 pi)]
    D9 --> D13[loglike]
    D10 --> D13
    D13 --> D14[where(isfinite, loglike, -inf)]
```

## Plain Text

```text
breakdown build_gp stage
└── scenario.build_gp(x)
    ├── quasisep_cpu
    │   └── tinygp.kernels.quasisep.Exp(sigma, scale)
    ├── quasisep_matern32_cpu
    │   └── tinygp.kernels.quasisep.Matern32(sigma, scale)
    ├── quasisep_matern52_cpu
    │   └── tinygp.kernels.quasisep.Matern52(sigma, scale)
    └── tinygp.GaussianProcess(kernel, X, diag=jitter^2, assume_sorted=True)
        ├── jax.vmap(mean_function)(X)
        ├── Diagonal(diag=broadcast(diag, mean.shape))
        └── QuasisepSolver(kernel, X, noise)
            ├── kernel.to_symm_qsm(X)
            │   ├── Pinf = kernel.stationary_covariance()
            │   ├── a = jax.vmap(kernel.transition_matrix)(append(X[0], X[:-1]), X)
            │   ├── h = jax.vmap(kernel.observation_model)(X)
            │   ├── q = h
            │   ├── p = h @ Pinf
            │   ├── d = jnp.sum(p * q, axis=1)
            │   ├── p = jax.vmap(lambda x, y: x @ y)(p, a)
            │   └── SymmQSM(
            │       diag=DiagQSM(d),
            │       lower=StrictLowerTriQSM(p, q, a),
            │   )
            ├── noise.to_qsm()
            │   └── Diagonal.to_qsm() -> DiagQSM(d=noise.diag)
            ├── matrix = kernel_qsm + noise_qsm
            │   └── SymmQSM.__add__
            │       └── ops.elementwise_add
            │           ├── deconstruct(a), deconstruct(b)
            │           ├── add_two(diag_a, diag_b)
            │           ├── add_two(lower_a, lower_b)
            │           ├── add_two(upper_a, upper_b)
            │           └── construct(...)
            └── factor = matrix.cholesky()
                └── SymmQSM.cholesky()
                    └── jax.lax.scan(impl, ...)
                        ├── ck = sqrt(dk - pk @ fp @ pk)
                        ├── tmp = fp @ ak.T
                        ├── wk = (qk - pk @ tmp) / ck
                        ├── fk = ak @ tmp + outer(wk, wk)
                        └── LowerTriQSM(
                            diag=DiagQSM(c),
                            lower=StrictLowerTriQSM(p=p, q=w, a=a),
                        )

runtime log_probability path
└── gp.log_probability(y)
    ├── gp._get_alpha(y)
    │   ├── y - gp.loc
    │   └── solver.solve_triangular(y - gp.loc)
    │       └── factor.solve(y - gp.loc)
    │           └── LowerTriQSM.solve()
    │               └── jax.lax.scan(impl, ...)
    │                   ├── xn = (yn - pn @ fn) / cn
    │                   ├── fn_next = an @ fn + outer(wn, xn)
    │                   └── alpha
    └── gp._compute_log_prob(alpha)
        ├── -0.5 * jnp.sum(jnp.square(alpha))
        ├── solver.normalization()
        │   ├── jnp.sum(jnp.log(factor.diag.d))
        │   └── 0.5 * N * log(2 pi)
        └── jnp.where(isfinite(loglike), loglike, -inf)
```

## Kernel-Specific Leaf Calls

These are the lowest kernel-specific methods reached by `build_gp(x)` through
`kernel.to_symm_qsm(X)`.

### `Exp`

```text
stationary_covariance() -> [[1]]
observation_model(X) -> [sigma]
transition_matrix(X1, X2)
└── dt = X2 - X1
└── exp(-dt / scale)
```

### `Matern32`

```text
stationary_covariance() -> diag([1, 3 / scale^2])
observation_model(X) -> [sigma, 0]
transition_matrix(X1, X2)
└── dt = X2 - X1
└── f = sqrt(3) / scale
└── exp(-f * dt) * [[1 + f dt, -f^2 dt], [dt, 1 - f dt]]
```

### `Matern52`

```text
stationary_covariance()
└── f = sqrt(5) / scale
└── [[1, 0, -f^2 / 3], [0, f^2 / 3, 0], [-f^2 / 3, 0, f^4]]

observation_model(X) -> [sigma, 0, 0]

transition_matrix(X1, X2)
└── dt = X2 - X1
└── f = sqrt(5) / scale
└── exp(-f * dt) * 3x3 polynomial transition matrix
```

## Benchmarked Stages

- `build_gp`
- `log_probability`
- `_get_alpha`
- `solver.solve_triangular`
- `factor.solve`
- `_compute_log_prob`
- `solver.normalization`
