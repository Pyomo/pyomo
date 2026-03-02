# optimize_experiments() Test Script Classification

## Goal
Keep `optimize_experiments()` tests aligned with file intent:
- `test_doe_build.py`: structure, schema, model expression/build invariants
- `test_doe_errors.py`: exceptions and warning behavior
- `test_doe_solve.py`: numerical algorithm/value checks

## Classification Rules
- Build/structure: verifies payload keys, timing field consistency, static helper/set contents, logging that indicates model structure application.
- Errors/warnings: verifies raised exceptions or warning messages for invalid/misconfigured usage.
- Solve/numerical: verifies selected points, objective values, FIM/objective numerical equivalence, serial-vs-parallel numerical equivalence.

## Warning Policy
- Default: warning assertions belong in `test_doe_errors.py`.
- Reason: warnings are API-behavior diagnostics for invalid, ambiguous, or risky usage, which is closest to error-handling intent.
- Exception (use sparingly): if a warning strictly validates build-time structural instrumentation (not user misconfiguration), it may live in `test_doe_build.py`.

## Moves Applied

### Moved to `test_doe_errors.py`
- `test_optimize_experiments_lhs_seed_requires_integer`
- `test_optimize_experiments_sym_break_var_must_be_input`
- `test_optimize_experiments_symmetry_mapping_failure_raises`
- `test_optimize_experiments_symmetry_breaking_default_variable_warning`
- `test_optimize_experiments_symmetry_breaking_multiple_markers_warning`
- `test_lhs_initialization_large_space_emits_warnings`

### Moved to `test_doe_build.py`
- `test_optimize_experiments_writes_results_file`
- `test_optimize_experiments_timing_includes_lhs_phase_separately`
- `test_optimize_experiments_symmetry_log_once_per_scenario`
- `test_optimize_experiments_lhs_diagnostics_populated`
- `test_maximize_objective_set_contents`
- `test_symmetrize_lower_tri_helper`

### Removed from `test_doe_solve.py`
- Error/warning/structure tests above, leaving solve focused on numerical behavior.
- Removed duplicate unsupported-parameter-scenarios coverage from solve; covered in `test_doe_errors.py`.

## Current `optimize_experiments()` Coverage by Intent

### `test_doe_solve.py` (numerical)
- Objective evaluation numerics and fallback semantics.
- FIM-at-point behavior with prior restoration.
- LHS chosen-combination matches brute-force oracle.
- Re-entrant call behavior on same object (light runtime behavior).
- Parallel/serial equivalence for LHS FIM eval and combo scoring.
- Timeout path with partial FIM evaluation.
- Determinant objective expected design values.

### `test_doe_errors.py` (exceptions + warnings)
- Invalid initialization method, invalid `lhs_n_samples`, invalid `lhs_seed` type.
- Invalid `n_exp`, invalid `results_file` type.
- Unsupported parameter scenarios.
- Invalid symmetry variable marker and mapping failure path.
- Warning behavior for missing/multiple symmetry markers.
- Warning behavior for very large LHS candidate/combo space.

### `test_doe_build.py` (structure)
- Experiment input variable discovery (direct + fd fallback).
- Multi-experiment model/result structure assertions.
- Results file payload schema keys.
- Timing decomposition consistency.
- LHS diagnostics structure and basic value semantics.
- Objective helper/static structure checks (`_MAXIMIZE_OBJECTIVES`, lower-tri symmetrization).
- Informational symmetry logging cardinality (applied once per scenario).

## Status
- [x] Reclassification implemented
- [x] Targeted test runs green
- [x] Full `pyomo/contrib/doe/tests` run green
