# Symbolic DoE PR Notes

This note is intended to help reviewers understand the scope of the symbolic
Pyomo.DoE pull request, the mathematical background for the main changes, and
how those ideas map onto the implementation.

## Overview

This PR adds symbolic-gradient support to `pyomo.contrib.doe` on top of the
current DoE implementation while preserving the newer objective and GreyBox
structure already present on `main`.

At a high level, the PR:

- adds symbolic / `pynumero` gradient support alongside the existing
  finite-difference workflow
- adds `ExperimentGradients` to organize the derivative information used by DoE
- keeps the newer DoE objective and GreyBox structure from current `main`
- adds a lightweight public `PolynomialExperiment` example for symbolic testing
- broadens test coverage for symbolic/automatic derivative consistency and DoE
  regression behavior

The PR does **not** change the underlying Fisher information matrix mathematics.
The main change is how the derivative information needed for those calculations
can be assembled.

## Mathematical Background

For a model with outputs `y` and unknown parameters `theta`, DoE needs the
output sensitivity matrix

```math
Q = \frac{\partial y}{\partial \theta}
```

to build the Fisher information matrix

```math
\mathrm{FIM} = Q^T \Sigma^{-1} Q
```

where `Sigma` is the measurement-error covariance matrix.

For models defined implicitly by equations

```math
F(x, u, \theta) = 0
```

the state sensitivities follow from differentiating the implicit system:

```math
\frac{\partial F}{\partial x}\frac{\partial x}{\partial \theta}
+
\frac{\partial F}{\partial \theta}
= 0
```

so

```math
\frac{\partial x}{\partial \theta}
=
-\left(\frac{\partial F}{\partial x}\right)^{-1}
\frac{\partial F}{\partial \theta}
```

Those state sensitivities are then propagated to the experiment outputs to form
`Q`, and then to the FIM.

The symbolic path in this PR changes how those Jacobian terms are assembled; it
does not change the equations above.

## Implementation Mapping

### `gradient_method`

The symbolic behavior in this PR is selected through `gradient_method`.

- finite-difference setup still uses `fd_formula`
- symbolic / `pynumero` behavior is selected by `gradient_method`
- `fd_formula="central"` remains the shared setup convention in many tests even
  when `gradient_method="pynumero"` is used

This means the symbolic path is not activated by changing the finite-difference
formula. Instead, the finite-difference configuration stays available while the
derivative backend is chosen independently.

### `ExperimentGradients`

`ExperimentGradients` is responsible for organizing the derivative information
used to build sensitivity matrices for DoE.

The important implementation detail in this PR is that the symbolic and
automatic derivative structures are prepared through a unified setup path. This
lets tests compare symbolic and automatic Jacobian entries directly while still
using the same overall experiment structure.

In other words, the PR moves the code toward:

- one shared setup path for Jacobian bookkeeping
- backend-specific derivative population within that structure

rather than maintaining more separate symbolic and automatic setup logic.

## Test Strategy

The test updates in this PR were aimed at checking both the mathematical path
and the practical DoE integration path.

### Polynomial

The polynomial example is the lightweight symbolic reference problem.

It is used for:

- exact gradient checks against hand-derived values
- symbolic vs automatic Jacobian consistency checks
- public example coverage
- generic 2D factorial / plotting coverage where a small two-design-variable
  example is helpful

Because the polynomial example has one output and four parameters, some FIM
regression tests use an identity prior to avoid rank-deficient raw FIMs when the
test purpose is metric regression rather than singular-matrix behavior.

### Rooney-Biegler

Rooney-Biegler is used for most of the general-purpose solve and regression
coverage because it is lightweight and still exercises the full DoE flow.

It is used for:

- symbolic `run_doe()` regression tests
- symbolic objective-matrix consistency checks
- bad-model / error-path coverage
- perturbed-point Jacobian agreement checks in the non-reactor replacements
- GreyBox helper and solve-path coverage

Several Rooney-Biegler determinant tests use a prior FIM. This is not meant to
change the underlying mathematics; it is there to keep the determinant-based
solve path well-conditioned for the small Rooney-Biegler problem.

### GreyBox

The non-solve GreyBox helper tests now use the Rooney-Biegler path as well.

Those tests were updated to match a two-parameter FIM instead of the older
reactor-specific four-parameter setup. In particular:

- the test FIM used for GreyBox helper checks is now `2 x 2`
- the reduced-Hessian finite-difference helper was generalized to use the
  current FIM dimension instead of assuming four parameters
- the build checks compare against the actual FIM carried by the GreyBox object
  rather than a hard-coded reactor-specific reconstruction

The solve-path checks still use the `cyipopt` GreyBox route and therefore remain
subject to the MA57/HSL runtime availability for that path.

## Notes And Caveats

### Reactor initialization nuance

One important nuance that came up during review is that the reactor model
required more care for generalized symbolic/automatic correctness checks.

In particular:

- the raw reactor labeled model is not always the cleanest reduced test vehicle
  for output-sensitivity checks
- some reactor-oriented checks were therefore replaced with lighter
  Rooney-Biegler or polynomial coverage where the test purpose was generic

### MA57 / HSL on the GreyBox `cyipopt` path

The GreyBox `cyipopt` path is distinct from simply having a working standalone
IPOPT executable. In local debugging, the relevant failure mode was that
`cyipopt` could be present while the MA57/HSL runtime needed on that path was
not available. The test skip message and code comment call that out explicitly
so it is not confused with generic IPOPT availability.

## Validation Snapshot

The final focused DoE test bundle used during this review pass was:

```bash
python -m pytest -q \
    pyomo/contrib/doe/tests/test_utils.py \
    pyomo/contrib/doe/tests/test_doe_solve.py \
    pyomo/contrib/doe/tests/test_doe_build.py \
    pyomo/contrib/doe/tests/test_doe_errors.py \
    pyomo/contrib/doe/tests/test_greybox.py
```

with local result:

```text
130 passed, 4 skipped, 5 warnings in 35.67s
```

The remaining warnings are the expected non-interactive matplotlib `Agg`
warnings from tests that call `draw_factorial_figure()`.
