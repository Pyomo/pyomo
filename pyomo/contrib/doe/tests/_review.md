# Review: `_DoEResultsJSONEncoder`, `_enum_label`, and `optimize_experiments()` API

**Reviewer**: Pyomo contributor review  
**Files**: `pyomo/contrib/doe/doe.py`, `tests/test_doe_build.py`, `tests/test_doe_errors.py`, `tests/test_doe_solve.py`  
**Date**: March 2026

---

## 1. `_DoEResultsJSONEncoder` — Is it needed?

**Yes, it is necessary and should be kept.**

The `self.results` dictionary populated by both `run_doe()` and
`optimize_experiments()` contains values that the standard `json.JSONEncoder`
cannot handle:

- `np.ndarray` — FIM matrices, sensitivity matrices, experiment design values
- `np.generic` (e.g. `np.float64`, `np.int64`) — scalar metric values
- `Enum` instances — `ObjectiveLib` and `FiniteDifferenceStep` values stored
  in `results["Objective expression"]` and `results["Finite Difference Scheme"]`

Without the custom encoder any `json.dump` call on `self.results` would raise
`TypeError` (which is exactly what `test_doe_results_json_encoder_unsupported_object_raises`
verifies for the unhandled case).

**Bug found**: `run_doe()` saves results with `json.dump(self.results, file)`
(bare, no `cls=` argument), while `optimize_experiments()` correctly uses
`cls=_DoEResultsJSONEncoder`. The `run_doe` save path will raise `TypeError` at
runtime for any experiment whose FIM cannot be natively serialized — which is
essentially every real experiment since `self.results["FIM"]` is an
`np.ndarray`. This should be fixed:

```python
# run_doe(), current (broken):
json.dump(self.results, file)

# run_doe(), correct (consistent with optimize_experiments):
json.dump(self.results, file, indent=2, cls=_DoEResultsJSONEncoder)
```

**Location**: The class lives at module level, outside `DesignOfExperiments`,
which is correct — it is not a class method. It could be moved to `utils.py`
to reduce the surface of `doe.py`, but both test files currently import it
directly from `doe.py` (`from pyomo.contrib.doe.doe import ...,
_DoEResultsJSONEncoder`), so any move would be an API break that requires
coordinated test updates.

---

## 2. `_enum_label` — Placement

`_enum_label` has been moved to immediately after `__init__`, before the "action"
methods (`run_doe`, `optimize_experiments`, etc.). The previous placement — before
`__init__` at the top of the class body alongside `_MAXIMIZE_OBJECTIVES` — was
non-standard: class-body constants belong above `__init__`, but methods,
including static utility methods, conventionally follow the constructor. The new
ordering (`__init__` → `_enum_label` → `run_doe` → `optimize_experiments` → ...)
matches Pyomo conventions.

---

## 3. `optimize_experiments()` API Review

### 3.1 Signature Explosion (12 Parameters)

```python
def optimize_experiments(
    self,
    results_file=None,
    n_exp: int = None,
    init_method=None,
    init_n_samples: int = 5,
    init_seed: int = None,
    init_parallel: bool = False,
    init_combo_parallel: bool = False,
    init_n_workers: int = None,
    init_combo_chunk_size: int = 5000,
    init_combo_parallel_threshold: int = 20000,
    init_max_wall_clock_time: float = None,
):
```

Eight of the twelve parameters exist solely to configure the LHS initialization
(`init_*`). These should be grouped into a single `lhs_options` keyword
argument that accepts a dict or a dedicated `LHSOptions` dataclass. This would
reduce the signature to 4 meaningful parameters and make it immediately clear
which options belong together:

```python
def optimize_experiments(
    self,
    results_file=None,
    n_exp: int = None,
    init_method=None,
    init_options: dict = None,   # replaces 8 init_* kwargs
):
```

Until this refactor is done, all `init_*` parameters should be checked even
when `init_method is None` — currently they are silently ignored if
`init_method` is not `"lhs"`, making it easy to pass `init_n_samples=100` and
get no initialization improvement and no error.

### 3.2 Implicit Template Mode vs. User-Initialized Mode

The method silently selects between two fundamentally different operating
modes based on `len(experiment_list)`:

- `len == 1` → **template mode**: clones the single experiment `n_exp` times
- `len > 1` → **user-initialized mode**: each item in the list is a distinct
  pre-configured experiment; `n_exp` must not be set

This mode-switching behavior is not communicated in the method signature or
its first-line docstring. A user who passes a one-element list expecting
user-initialized mode (because they want only that one experiment) is silently
placed in template mode. Explicit mode selection would be clearer:

```python
def optimize_experiments(
    self,
    results_file=None,
    n_exp: int = None,
    mode: str = "auto",   # "template" | "user_initialized" | "auto"
    ...
):
```

Or at minimum, document the mode-detection rule prominently in the docstring
(currently it is only described inside the "Notes" section's "Number of
Experiments" sub-heading).

### 3.3 `n_param_scenarios` Is Hardcoded to 1

```python
n_param_scenarios = 1  # currently single-scenario optimization
```

The entire `param_scenario_blocks` nesting (`self.model.param_scenario_blocks[s].exp_blocks[k]`)
exists to support parametric uncertainty scenarios, but this is guarded by a
TODO comment and `n_param_scenarios` is always 1. The nesting adds substantial
complexity to the model structure, results schema, symmetry breaking code, and
tests — all for a feature that does not yet exist.

Options:
1. **Implement parametric scenarios** or
2. **Remove the nesting** and replace `param_scenario_blocks[0].exp_blocks[k]`
   with flat `exp_blocks[k]` until the feature is ready.

Keeping placeholder scaffolding that is never activated in production increases
maintenance cost and increases the model's variable/constraint count slightly
for every solve.

### 3.4 Symmetry Breaking Is an Implicit Side Effect

When `n_exp > 1`, the method automatically adds ordering constraints between
experiments using the first experiment input if the user has not marked a
`sym_break_cons` Suffix. This is a semantically significant modification of the
optimization problem that happens silently:

```python
# Added without user request:
self.model.param_scenario_blocks[s].add_component(
    f"symmetry_breaking_s{s}_exp{k}",
    pyo.Constraint(expr=var_prev <= var_curr)
)
```

The automatic symmetry breaking:
1. Changes the feasible region (excludes valid solutions where `exp[0].var > exp[1].var`)
2. Is not reflected in the returned `results` in any easily checkable flag
   (only in `results["diagnostics"]["symmetry_breaking"]["source"]` which is
   a nested dict key)
3. Uses the `self.logger.warning()` channel but the warning is not a
   `warnings.warn(...)` that could be caught and suppressed by the caller

**Recommendation**: Make symmetry breaking opt-in via an explicit parameter
(`add_symmetry_breaking: bool = True`) or at minimum document it as ON by
default in the signature-level docstring, not only in the Notes section.

### 3.5 Dual Results Schema: Flat Dictionary and Structured Payload

`self.results` contains both a flat set of keys (backwards-compatible with
`run_doe`):

```python
self.results["Solver Status"]        # flat
self.results["FIM"]                  # flat (run_doe compat)
```

and a structured nested payload added in the refactor:

```python
self.results["run_info"]["solver"]["status"]     # structured
self.results["settings"]["objective"]["name"]    # structured
self.results["scenarios"][0]["experiments"][0]   # structured
```

These two schemas must be kept in sync. Currently they are not fully consistent:
- `results["Solver Status"]` is the raw `SolverStatus` enum;
  `results["run_info"]["solver"]["status"]` is the same value via
  `self.results["Solver Status"]`.
- `results["FIM"]` is absent from `optimize_experiments` output (the FIM lives
  inside `results["Scenarios"][0]["Experiments"][0]["FIM"]`), creating
  asymmetry with `run_doe()` output where `results["FIM"]` is top-level.

**Recommendation**: Deprecate the flat top-level keys and make the structured
schema primary. Provide a migration guide for users currently consuming the flat
dict.

### 3.6 `_safe_metric` Is a Nested Function That Should Be a Method

```python
def _safe_metric(metric_name, compute_fn, scenario_index):
    try:
        val = float(compute_fn())
        return val if np.isfinite(val) else float("nan")
    except Exception as exc:
        self.logger.warning(...)
        return float("nan")
```

This captures `self.logger` from the enclosing method. Nested functions that
capture `self` are equivalent to instance methods and should be defined as such
(or as a module-level utility) to allow independent testing and re-use.

### 3.7 `run_doe` Result Schema Differs From `optimize_experiments` Result Schema

The two primary solve methods produce incompatible result objects:

| Key | `run_doe` | `optimize_experiments` |
|---|---|---|
| `"FIM"` | top-level `np.ndarray` | inside `"Scenarios"[0]["Experiments"][k]["FIM"]` |
| `"Sensitivity Matrix"` | top-level | per-experiment inside `"Scenarios"` |
| `"Experiment Design"` | top-level list | per-experiment inside `"Scenarios"` |
| `"run_info"` | absent | present |
| `"scenarios"` | absent | present |

Users who want to switch from `run_doe` to `optimize_experiments(n_exp=1)` must
update all result-access code. The two methods should produce the same first-level
keys for overlapping concepts.

### 3.8 `results_file` Validation Is Not Tested for `optimize_experiments`

`run_doe()` has `test_reactor_check_results_file_name` in `test_doe_errors.py`
that passes `results_file=int(15)` and expects a `ValueError`. The identical
validation exists in `optimize_experiments()` but has no corresponding test.
The missing test should be added to `TestDoEErrors`.

### 3.9 Silent Discard of Solver Failure

When the NLP solver terminates sub-optimally (infeasible, user-interrupted,
numerical difficulties), `optimize_experiments()` stores the result dict exactly
as if the solve succeeded. There is no warning or exception raised. In a
non-interactive workflow the user must remember to check
`results["Solver Status"]`. A minimum safeguard would be:

```python
if not pyo.check_optimal_termination(res):
    self.logger.warning(
        "optimize_experiments: solver terminated with status '%s'. "
        "Results may be unreliable.",
        res.solver.termination_condition,
    )
```

The same issue exists in `run_doe()`.

### 3.10 `_DoEResultsJSONEncoder` Not Applied in `run_doe`

As noted in Section 1, `run_doe()` calls `json.dump(self.results, file)` without
`cls=_DoEResultsJSONEncoder`. This will raise `TypeError` at runtime when
`results_file` is specified. The fix is trivial — apply the encoder — but
the bug is undetected because no test in the suite runs `run_doe` with a
`results_file` argument.

---

## 4. Test Coverage Issues for `optimize_experiments()`

*(For test-file structural issues see the existing `_test_review.md`. This section
focuses on gaps specific to `optimize_experiments`.)*

### 4.1 `results_file` Validation Not Tested for `optimize_experiments`

`test_reactor_check_results_file_name` in `test_doe_errors.py` tests the
`run_doe` path. No equivalent test exists for `optimize_experiments`.  
**Fix**: Add `test_optimize_experiments_results_file_invalid_type`.

### 4.2 `run_doe` + `results_file` Bug Is Not Caught by Any Test

The `run_doe` `json.dump` bug (Section 3.10) is undetected because no test
calls `run_doe(results_file=...)`. There is an analogous test for
`optimize_experiments` (`test_optimize_experiments_writes_results_file`).  
**Fix**: Add `test_run_doe_writes_results_file`.

### 4.3 `test_optimize_experiments_init_method_enum_accepted` and `test_optimize_experiments_init_method_enum_invalid_init_n_samples` Are Identical

Both tests in `test_doe_errors.py` assert the same exception message with the
same call:

```python
# test_optimize_experiments_init_method_enum_accepted
doe_obj.optimize_experiments(init_method=InitializationMethod.lhs, init_n_samples=0)
# →  ValueError: "``init_n_samples`` must be a positive integer, got 0."

# test_optimize_experiments_init_method_enum_invalid_init_n_samples
doe_obj.optimize_experiments(init_method=InitializationMethod.lhs, init_n_samples=0)
# →  same assertion
```

These are duplicate tests. The first test name claims to verify that the
`InitializationMethod` enum is accepted. It does not: the enum would be accepted
even if the call path were wrong, because the `ValueError` fires *after* enum
parsing. Neither test asserts that `InitializationMethod.lhs` is accepted when
all other parameters are valid.  
**Fix**: Remove the duplicate. Add a test that passes valid `init_n_samples`
with the enum value and confirms no `ValueError` is raised (see also
`_test_review.md` C3).

### 4.4 No Test for `results_file` With `optimize_experiments` + `run_doe` Schema Difference

There is no test that calls `optimize_experiments(results_file=...)`, reads the
written file, and asserts that keys specific to `optimize_experiments`
(`"Scenarios"`, `"run_info"`, etc.) are present while `run_doe`-only keys (top-
level `"FIM"`, `"Sensitivity Matrix"`) are absent. The existing
`test_optimize_experiments_writes_results_file` checks for the presence of the
new structured keys but does not verify the schema boundary.

### 4.5 No Test That `optimize_experiments` With `n_exp=1` in Template Mode Produces FIM Consistent With `run_doe`

A single-experiment call to `optimize_experiments(n_exp=1)` should produce the
same optimal FIM as `run_doe()` on the same problem with the same objective.
This cross-method consistency property is untested. If the model construction
for `optimize_experiments` introduces an off-by-one error in the prior FIM or
the aggregation logic, it would not be caught by any current test.

### 4.6 No Test for `optimize_experiments` With `results_file` and Non-Serializable Result Value

`test_doe_results_json_encoder_unsupported_object_raises` validates the encoder's
`super().default()` fallback path directly. There is no integration test that
places a non-standard object in `self.results` and confirms that writing to a
file raises a descriptive error rather than silently writing a corrupt file.

---

## 5. Summary of Action Items

| Priority | Location | Item |
|---|---|---|
| **High** | `doe.py` | Fix `run_doe` to use `cls=_DoEResultsJSONEncoder` in `json.dump` |
| **High** | `doe.py` | Document or make explicit the template vs. user-initialized mode selection |
| **High** | `doe.py` | Add deprecation or loud warning on sub-optimal solver termination |
| **High** | `test_doe_errors.py` | Add `test_optimize_experiments_results_file_invalid_type` |
| **High** | `test_doe_build.py` | Add `test_run_doe_writes_results_file` to catch the missing-encoder bug |
| Medium | `doe.py` | Group LHS parameters into `lhs_options` dict or `LHSOptions` dataclass |
| Medium | `doe.py` | Make symmetry breaking opt-in or document at signature level |
| Medium | `doe.py` | Remove `n_param_scenarios` scaffolding until parametric scenario feature is implemented |
| Medium | `doe.py` | Promote `_safe_metric` to a private instance method |
| Medium | `doe.py` | Unify `run_doe` and `optimize_experiments` result schemas |
| Medium | `test_doe_errors.py` | Remove duplicate `test_optimize_experiments_init_method_enum_*` tests |
| Low | `doe.py` | Move `_DoEResultsJSONEncoder` to `utils.py` (coordinate test import updates) |
| Low | `test_doe_build.py` | Add cross-method FIM consistency test (`run_doe` vs `optimize_experiments(n_exp=1)`) |
