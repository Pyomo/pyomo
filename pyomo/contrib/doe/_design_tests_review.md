# Critical Design & Test Review: `optimize_experiments()` API

> Reviewer analysis of `pyomo/contrib/doe/doe.py` and
> `pyomo/contrib/doe/tests/` as of March 2026.

---

## Part 1 — API Design Issues

### 1. Signature Explosion via Flat LHS Sub-Options

`optimize_experiments()` exposes 14 parameters, 8 of which are exclusively
LHS-initialization plumbing:

```python
def optimize_experiments(
    self,
    parameter_scenarios=None,
    results_file=None,
    n_exp: int = None,
    initialization_method=None,
    lhs_n_samples: int = 5,          # LHS-only
    lhs_seed: int = None,             # LHS-only
    lhs_parallel: bool = False,       # LHS-only
    lhs_combo_parallel: bool = False, # LHS-only
    lhs_n_workers: int = None,        # LHS-only
    lhs_combo_chunk_size: int = 5000, # LHS-only
    lhs_combo_parallel_threshold: int = 20000,  # LHS-only
    lhs_max_wall_clock_time: float = None,       # LHS-only
):
```

Every future initialization strategy (e.g., random restart, Bayesian) will add
another batch of top-level parameters. **Better design**: group LHS options into
a dedicated config object passed as a single argument.

```python
# Suggested
@dataclass
class LHSOptions:
    n_samples: int = 5
    seed: int | None = None
    parallel: bool = False
    combo_parallel: bool = False
    n_workers: int | None = None
    combo_chunk_size: int = 5000
    combo_parallel_threshold: int = 20_000
    max_wall_clock_time: float | None = None

def optimize_experiments(
    self,
    n_exp: int = None,
    initialization_options=None,   # None | LHSOptions | ...
    results_file=None,
    parameter_scenarios=None,
):
```

This pattern mirrors how Pyomo solver options work and is easier to document,
validate, and extend.

---

### 2. `initialization_method` Should Be an Enum, Not a Raw String

The rest of the codebase uses `Enum` for bounded string options
(`ObjectiveLib`, `FiniteDifferenceStep`). `initialization_method` breaks that
convention by accepting a plain string:

```python
if initialization_method not in (None, "lhs"):
    raise ValueError(...)
```

A string is not discoverable in IDEs, can be mistyped silently if the guard is
moved, and is harder to type-check. A companion `InitializationMethod` enum (or
extending the config-object approach above) would be consistent.

---

### 3. `parameter_scenarios` Is a Dead Placeholder in the Public Signature

`parameter_scenarios` is documented as *"currently unsupported; passing
anything other than `None` raises `NotImplementedError`"*. Placing an
unimplemented parameter in the public signature:

- Confuses users scanning the method signature or generated docs.
- Creates a regression risk: any future refactor that accidentally passes it
  will get a confusing runtime error.
- Has no automated test that demonstrates a complete path through it.

**Recommendation**: Remove it from the public signature and add an underscore-
prefixed private keyword (`_parameter_scenarios=None`) or a plain comment until
the feature is ready.

---

### 4. Solver Failure After Square Solve Is Not Checked

The initialization (square) solve is fire-and-forget:

```python
# doe.py ~line 1074
self.model.dummy_obj = pyo.Objective(expr=0, sense=pyo.minimize)
self.solver.solve(self.model, tee=self.tee)   # ← result never inspected
```

If IPOPT returns infeasible or hits a wall-clock limit here, the subsequent full
optimization starts from garbage initial values with no warning to the user.
The commented TODO block in `run_doe()` (lines 406-414) acknowledges the same
gap. Both methods should inspect the solver result and at minimum emit a
`logger.warning` (or raise, depending on severity policy).

**Suggested fix**:

```python
res_init = self.solver.solve(self.model, tee=self.tee)
if not pyo.check_optimal_termination(res_init):
    self.logger.warning(
        "Square-solve initialization did not terminate optimally "
        f"({res_init.solver.termination_condition}). "
        "Proceeding but solution quality may be poor."
    )
```

---

### 5. Prior FIM Is Read Late, Not Snapshotted at Construction

`self.prior_FIM` is stored at construction time but read throughout
`optimize_experiments()` at solve time. A user who mutates `doe_obj.prior_FIM`
between construction and `optimize_experiments()` silently changes the problem.
This behavior is undocumented.

**Options**:
- Document explicitly that `prior_FIM` is mutable state and mutations are
  honored.
- Or make a defensive copy at the start of `optimize_experiments()` (like
  `_lhs_initialize_experiments` already does: `prior = self.prior_FIM.copy()`).

---

### 6. Duplicate / Inconsistent Result Dictionary Keys

`self.results` is built with two parallel naming conventions:

| Old (human-readable, space-separated) | New (snake_case structured) |
|---|---|
| `"Solver Status"` | `results["run_info"]["solver"]["status"]` |
| `"Wall-clock Time"` | `results["timing"]["total_s"]` |
| `"Scenarios"][0]["Total FIM"]` | `results["scenarios"][0]["total_fim"]` |
| `"Number of Experiments per Scenario"` | `results["settings"]["modeling"]["n_experiments_per_scenario"]` |

Both forms are written to the serialized JSON, doubling the payload size and
creating two surfaces to keep in sync. A single canonical form (the structured
`run_info / settings / timing / scenarios` hierarchy) with backwards-compat
aliases (or a migration notice) is cleaner.

---

### 7. Template Mode vs. User-Initialized Mode Not Surfaced in Results

The internal `_template_mode` boolean is a local variable that is never
included in `self.results`. Post-hoc inspection of a saved JSON file cannot
determine whether the run was template-mode or user-initialized-mode. Adding
`"template_mode": _template_mode` to `results["settings"]["modeling"]` costs
nothing.

---

### 8. Silent Fall-Through When `lhs_combo_parallel=True` but Threshold Not Met

The docstring for `lhs_combo_parallel` says:

> *"The flag has no effect unless … the total number of combinations exceeds
> `lhs_combo_parallel_threshold`."*

But there is no feedback to the user when parallelism is silently ignored:

```python
use_parallel_combo = (
    lhs_combo_parallel
    and n_combinations >= lhs_combo_parallel_threshold
    and resolved_workers > 1
)
```

A user who passes `lhs_combo_parallel=True` but has fewer combinations than the
threshold will get serial execution with no indication. The diagnostics dict
includes `combo_mode` which reveals the truth, but only if the user actively
inspects it. A `logger.info` at decision time would be helpful.

---

### 9. Symmetry Breaking Variable Detection Is Fragile in Edge Cases

Symmetry breaking is inferred from `fd_scenario_blocks[0]` of the first
experiment block. Two edge cases are unguarded:

1. **Multiple Suffix markers**: If the user marks more than one variable in
   `sym_break_cons`, only the first is used and a warning is issued. However,
   the warning message names the variable by `.name` (full hierarchical path),
   which can be long and unclear. The diagnostics dict stores the `.local_name`.
   These should be consistent.

2. **Auto-selection silently constrains feasibility**: When no `sym_break_cons`
   Suffix is provided, the auto-selected variable is the first element returned
   by `iter(first_exp_block.experiment_inputs)`. Python dict iteration order is
   insertion-ordered since 3.7, but this is implementation-dependent on the
   user's `get_labeled_model` labeling order. Users may not realize their
   feasible region has been constrained.

---

## Part 2 — Test Issues

### T1. `get_standard_args()` Uses the Deprecated `experiment` Kwarg

In `test_doe_solve.py`:

```python
def get_standard_args(experiment, fd_method, obj_used):
    args = {}
    args['experiment'] = experiment   # ← deprecated kwarg
    ...
```

Every test that calls `get_standard_args` exercises the deprecated path,
exercising the deprecation-warning code path rather than the supported API. This
unnecessarily inflates the coverage of dead code while obscuring gaps against
the `experiment_list` path. The helper should be updated to:

```python
args['experiment_list'] = [experiment]
```

---

### T2. Integration Tests Require IPOPT; No Mocked Unit Tests for `optimize_experiments` internals

All meaningful `optimize_experiments` tests carry:

```python
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
```

The non-solver portions of the method – model building, result dict
construction, symmetry constraint injection, prior FIM initialization – are only
testable on machines where IPOPT is installed. This is fragile for CI pipelines
and makes the test run non-deterministic on solver availability.

**Recommendation**: Mock the solver call in selected unit tests:

```python
with patch.object(doe_obj.solver, "solve", return_value=_mock_ok_result()):
    doe_obj.optimize_experiments(n_exp=2)
self.assertIn("Total FIM", doe_obj.results["Scenarios"][0])
```

This pattern already works for `_lhs_initialize_experiments` tests, which use
`patch.object(doe, "_compute_fim_at_point_no_prior", ...)`.

---

### T3. Hardcoded Solver-Specific Numerical Values in `test_optimize_experiments_determinant_expected_values`

```python
expected_hours = [1.9321985035514362, 9.999999685577139]
self.assertAlmostEqual(scenario["log10 D-opt"], 6.028152580313302, places=3)
```

These constants are IPOPT-specific convergence artifacts. A change in solver
tolerance, IPOPT version, platform, or MA57 variant can cause false regression
failures. **Better**: assert qualitative properties that hold for any converged
solution:

```python
# Symmetry constraint is satisfied
self.assertLessEqual(got_hours[0], got_hours[1])
# Design is at a known optima boundary (within tolerance of upper bound)
self.assertAlmostEqual(max(got_hours), HOUR_UPPER_BOUND, places=1)
# FIM is positive definite
total_fim = np.array(scenario["Total FIM"])
self.assertTrue(np.all(np.linalg.eigvalsh(total_fim) > 0))
# Objective improved relative to a known baseline
self.assertGreater(scenario["log10 D-opt"], KNOWN_LOWER_BOUND)
```

---

### T4. `test_optimize_experiments_is_reentrant_on_same_object` Only Checks Schema, Not Correctness

```python
self.assertEqual(len(first_design), len(second_design))
self.assertIn("timing", doe.results)
```

This does not verify that:
- The two calls produce the same or compatible optimal design when the problem is
  deterministic.
- The model is truly rebuilt (not carrying over stale components from the first
  call).
- No Pyomo component naming collision occurs on the second call.

A stronger version would check that `doe.model.param_scenario_blocks` has the
correct cardinality after the second call, and that the timing entries from the
*second* call are present (not the first call's values).

---

### T5. LHS Tests Cover Only Single-Input, Two-Experiment Case

`test_optimize_experiments_lhs_matches_bruteforce_combo` uses a model with one
experiment input (`hour`) and `n_exp=2`. The marginal LHS decomposition strategy
is only meaningful for multi-input (multi-dimensional) problems – decomposing
into per-dimension samples and taking the Cartesian product only offers Latin
Hypercube properties in each dimension independently. There is no test with
`n_inputs > 1`, which means the per-dimension seeding logic in
`_lhs_initialize_experiments` is untested for the multi-input case.

---

### T6. No Test Verifying That LHS Initialization Actually Improves Final Solution Quality

The test suite verifies that:
- LHS picks the theoretically best candidate combo (brute-force match).
- The chosen initial points are injected into the model variables before solve.

But there is **no test** that the final NLP objective after optimization is
better (or at least not worse) when using `initialization_method="lhs"` vs. the
default initialization. This is the core claim of the feature and remains
unvalidated.

A test could be constructed on a bi-modal synthetic problem (or using a
well-known local sub-optimal starting point) where LHS reliably avoids the bad
local minimum.

---

### T7. `results_file` Test Does Not Exercise the `pathlib.Path` Branch

```python
doe_obj.optimize_experiments(n_exp=1, results_file=results_path)  # str
```

The API accepts both `str` and `pathlib.Path`. Only the string form is tested.
A one-line addition exercises the Path branch:

```python
doe_obj.optimize_experiments(n_exp=1, results_file=pathlib.Path(results_path))
```

---

### T8. Symmetry Breaking Warning Tests Are Mixed With Non-Solver Tests in the Same Class

`TestDoEErrors` contains a mix of:
- Pure input-validation tests (no solver needed, no `skipIf`).
- Tests that build and solve a model, requiring IPOPT
  (`test_optimize_experiments_sym_break_var_must_be_input`,
  `test_optimize_experiments_symmetry_mapping_failure_raises`, etc.).

The per-test `@unittest.skipIf(not ipopt_available, ...)` decorators work, but
the class name `TestDoEErrors` is misleading for solver-dependent tests.
**Recommendation**: extract solver-dependent error tests into a separate class,
e.g., `TestDoEErrorsRequiringSolver`, or move them to `test_doe_solve.py`.

---

### T9. `test_optimize_experiments_lhs_diagnostics_populated` Does Not Verify LHS Quality

The test fires `optimize_experiments` with all LHS options enabled and checks
structural keys. It asserts `best_obj > 0` but does not check that the reported
`best_obj` is actually the best achievable over the sampled candidates (which
would catch bugs in the combo scoring logic returning a stale or empty result).
Pairing this with a mock FIM and a hand-computed expected best would tighten the
test.

---

### T10. `lhs_combo_parallel_threshold` Boundary Condition Not Tested

`_lhs_initialize_experiments` uses:

```python
use_parallel_combo = (
    lhs_combo_parallel
    and n_combinations >= lhs_combo_parallel_threshold
    and resolved_workers > 1
)
```

There is no test that exercises `n_combinations == lhs_combo_parallel_threshold`
(boundary) or `n_combinations == lhs_combo_parallel_threshold - 1` (just below),
verifying that the serial vs. parallel switch fires correctly. The existing
`test_lhs_combo_parallel_matches_serial` forces parallelism by setting
`lhs_combo_parallel_threshold=1`, which always triggers parallel, never
exercising the threshold logic.

---

### T11. `n_exp=1` in User-Initialized Mode Is Not Tested

The code distinguishes between:
- **Template mode**: `len(experiment_list) == 1` → `n_exp` controls repetition.
- **User-initialized mode**: `len(experiment_list) > 1` → `n_exp` must not be
  set.

A user who passes `experiment_list=[single_exp]` always gets template mode.
There is no way to enter user-initialized mode with one experiment. This is not
a bug, but it is undocumented and untested as an intentional design decision. A
test or docstring note could clarify why a single-element list is treated as
template mode (and that `n_exp=1` is the default in that case).

---

### T12. Error Tests Construct Full `DesignOfExperiments` for Pure Argument-Validation Checks

```python
def test_optimize_experiments_invalid_initialization_method(self):
    doe_obj = DesignOfExperiments(
        experiment_list=[RooneyBieglerMultiExperiment(hour=2.0)],
        objective_option="pseudo_trace",
    )
    with self.assertRaisesRegex(ValueError, ...):
        doe_obj.optimize_experiments(initialization_method="bad")
```

The `RooneyBieglerMultiExperiment` constructor and `DesignOfExperiments.__init__`
are non-trivial. For pure argument-validation tests, a lightweight stub
experiment (one that does minimal work in `__init__`) would be faster and
isolate the test more cleanly. This also avoids the test breaking if
`RooneyBieglerMultiExperiment` itself changes.

---

## Summary Table

| # | Category | Severity | Item |
|---|---|---|---|
| 1 | API Design | High | 14-param signature; LHS options should be a config dataclass |
| 2 | API Design | Medium | `initialization_method` is a string; should be an `Enum` |
| 3 | API Design | Medium | `parameter_scenarios` dead placeholder in public signature |
| 4 | API Design | High | Square-solve result never checked; bad init silently proceeds |
| 5 | API Design | Low | `prior_FIM` mutation between calls is undocumented |
| 6 | API Design | Medium | Dual key-naming convention inflates and duplicates result dict |
| 7 | API Design | Low | `_template_mode` not surfaced in `self.results` |
| 8 | API Design | Low | Silent no-op when `lhs_combo_parallel=True` below threshold |
| 9 | API Design | Low | Auto symmetry-break variable choice undocumented; feasibility impact |
| T1 | Tests | High | `get_standard_args` uses deprecated `experiment` kwarg in all solve tests |
| T2 | Tests | High | No mocked unit tests for model-build / result-dict parts of `optimize_experiments` |
| T3 | Tests | Medium | Hardcoded solver-specific numerical expectations; fragile across platforms |
| T4 | Tests | Medium | Reentrance test only checks schema shape, not solution correctness |
| T5 | Tests | Medium | LHS brute-force match test only covers single-input, 2-experiment case |
| T6 | Tests | High | No test that LHS initialization actually improves final solution quality |
| T7 | Tests | Low | `results_file` test never exercises `pathlib.Path` input type |
| T8 | Tests | Low | Solver-dependent error tests mixed into `TestDoEErrors` class |
| T9 | Tests | Medium | LHS diagnostics test checks structure but not scoring fidelity |
| T10 | Tests | Medium | `lhs_combo_parallel_threshold` boundary never exercised |
| T11 | Tests | Low | Single-element `experiment_list` = template mode; undocumented and untested |
| T12 | Tests | Low | Full experiment object constructed just to test argument validation |
