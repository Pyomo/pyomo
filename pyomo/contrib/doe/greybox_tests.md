# Greybox Tests for `optimize_experiments()`

## Goal

The recent greybox support in `optimize_experiments()` is more than a new
solver path. It changes how the API builds the objective, initializes the
scenario-level FIM, and reports results. The most meaningful additional tests
should therefore focus on:

1. deterministic invariants of the aggregated FIM wiring,
2. initialization behavior before the final solve,
3. user-facing routing and result metadata,
4. a small number of end-to-end solve checks for supported objectives.

From a testing perspective, this gives us the best signal-to-noise ratio:

- Build-level tests are typically `O(1)` in solver calls and are much more
  stable across platforms.
- End-to-end solve tests are slower and can be numerically sensitive, so we
  should add them only where they validate behavior that build tests cannot.

## Current Coverage Snapshot

We already have useful coverage for:

- aggregated scenario FIM initialization into the greybox block,
- final solve routing through `grey_box_solver`,
- rejection of unsupported `pseudo_trace`,
- end-to-end solves for D-opt, A-opt, E-opt, and ME-opt.

The biggest remaining gaps are around prior-FIM handling, weighted scenarios,
reentrancy, LHS initialization, and deterministic validation of all four
greybox metrics on the aggregated multi-experiment FIM.

## Priority Legend

- `P0`: High-value regression tests. Add these first.
- `P1`: Important follow-up tests. Add after `P0`.
- `P2`: Nice-to-have coverage and polish.

## Recommended Tests

| Priority | Test | Why it matters | Suggested file |
| --- | --- | --- | --- |
| `P1` | `test_optimize_experiments_single_experiment_greybox_path_works` | Even though the recent work was driven by multi-experiment support, `optimize_experiments()` can still be used with `n_exp=1`. We should make sure the greybox path behaves correctly there and does not depend on multi-experiment-only assumptions. | `test_greybox.py` or `test_doe_solve.py` |
| `P1` | `test_optimize_experiments_greybox_initialization_refreshes_inputs_after_square_solve` | The greybox block is built before the final solve but should be re-seeded from the solved aggregated FIM after the square initialization. A test should verify that the post-square values differ from the initial build values when the design changes, proving that initialization is not stale. | `test_doe_build.py` |
| `P1` | `test_optimize_experiments_greybox_zero_objective_is_rejected` | We already reject `pseudo_trace`. The same user-facing validation should be tested explicitly for `zero`, since it is also outside the supported greybox metric set. | `test_doe_errors.py` |
| `P1` | `test_optimize_experiments_greybox_tee_flag_reaches_solver` | `grey_box_tee` is a user-facing option. A mock solver should verify that the final greybox solve receives the correct `tee` value. This is small, but it is exactly the kind of integration detail that quietly breaks. | `test_doe_build.py` |
| `P2` | `test_optimize_experiments_greybox_safe_metric_failure_sets_nan_or_user_visible_diagnostic` | The algebraic path already has safe-metric-failure coverage. A greybox-specific version would help ensure singular or ill-conditioned aggregated FIMs fail predictably from the user perspective instead of crashing with an opaque exception. | `test_doe_solve.py` |
| `P2` | `test_optimize_experiments_greybox_default_solver_creation_smoke_test` | When `use_grey_box_objective=True` and the user does not pass `grey_box_solver`, the constructor creates a default `cyipopt` solver. A smoke test would protect that convenience path. | `test_greybox.py` |
| `P2` | `test_optimize_experiments_greybox_result_payload_contains_solver_and_metric_metadata` | This would assert that `run_info`, `settings`, scenario metrics, and solver names are all populated consistently for greybox runs. It is more about API polish than core math, so it can come later. | `test_doe_build.py` |

## Recommended Order

1. Add deterministic build tests for all four metrics on aggregated `total_fim`.
2. Add prior-FIM and solver-routing tests for the greybox path.
3. Add the reentrant test on the same DoE object.
4. Add LHS coverage for E-opt and ME-opt.
5. Add weighted-scenario and single-experiment follow-up coverage.
6. Add polish tests for validation, tee propagation, and result payload details.

## Notes on Test Style

- Prefer deterministic assertions on `scenario.total_fim`, greybox inputs, and
  greybox outputs over asserting exact optimizer-selected design points.
- When solve tests are necessary, prefer checking:
  - sorted hour selections,
  - objective values within reasonable tolerance,
  - solver routing and result metadata,
  instead of exact unsorted designs that can vary by equivalent permutation.
- For greybox solve tests, use slightly looser tolerances than the algebraic
  path when the solver is known to settle at nearby equivalent solutions.
