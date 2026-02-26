# `optimize_experiments` — Comprehensive Code Review

This document is a reference review of `DesignOfExperiments.optimize_experiments()` in
`pyomo/contrib/doe/doe.py`.  It covers the method top-to-bottom: structure, data flow,
correctness issues, API concerns, and test gaps.

## Progress Checklist (Current Branch)

- [x] OE-2: guard `results_message` when solver message is neither `str` nor `bytes`
- [x] OE-5: fix missing `f` prefix in `lhs_parallel` validation error message
- [x] OE-8: guard FIM metric computation (`A/D/E/ME` and structured `condition_number`) against numerical failures
- [x] OE-3: add JSON-safe serialization for numpy/enum payloads
- [x] OE-4: make `optimize_experiments` re-entrant on same object
- [x] OE-9: separate LHS preprocessing time from square-solve initialization time
- [x] OE-13: validate `sym_break_cons` variable is an experiment input
- [x] OE-16: validate `lhs_seed` type (`None` or `int`)
- [x] OE-10: remove `dummy_obj` after square solve to avoid component reuse

---

## 1. Method Overview

`optimize_experiments` is the main entry point for multi-experiment Model-Based Design of
Experiments.  It does the following in sequence:

```
1. Validate arguments (n_exp, initialization_method, lhs_* params)
2. Build model: param_scenario_blocks → exp_blocks → create_doe_model per experiment
3. Add symmetry-breaking constraints between experiments
4. Create aggregated multi-experiment objective  (create_multi_experiment_objective_function)
5. [Optional] LHS initialization  (_lhs_initialize_experiments → set variable values)
6. Square solve  (fix experiment inputs, dummy obj=0, solve → init FIM/jacobian)
7. Initialize scenario-level variables  (L, L_inv, fim_inv, determinant, pseudo_trace)
8. Main NLP solve  (unfix inputs, activate objective, solve)
9. Collect and return results  (self.results dict)
10. [Optional] Save results to JSON
```

**Key data structures:**

| Object | Role |
|--------|------|
| `self.model.param_scenario_blocks[s]` | One per parameter scenario (currently always 1) |
| `.exp_blocks[k]` | One per experiment in scenario `s` |
| `.fd_scenario_blocks[0..2*n_params]` | FD perturbation scenarios per experiment |
| `scenario.total_fim[p,q]` | Aggregated FIM summed across experiments |
| `scenario.obj_cons` | Block holding L, determinant, pseudo_trace, objective constraints |
| `self.results` | Flat + structured dict; both legacy flat keys and nested `settings/timing/names/diagnostics` |

---

## 2. Argument Resolution and Operating Modes

### 2.1 Template mode vs user-initialized mode

```python
if n_list > 1:           # user-initialized: experiment_list already has all experiments
    n_exp = n_list
    _template_mode = False
else:                    # template mode: single experiment cloned n_exp times
    n_exp = n_exp or 1
    _template_mode = True
```

**What this means in practice:**

- **Template mode** (`len(experiment_list) == 1`): all `n_exp` experiment blocks are
  created with `experiment_index=0`.  The model is exactly the same for every block
  (same structure, same initial values).  The NLP solver finds the best `n_exp`
  distinct experiment designs.
- **User-initialized mode** (`len(experiment_list) > 1`): experiment block `k` uses
  `experiment_index=k`.  Each block starts from a user-supplied model with potentially
  different nominal values.

**Issue — LHS initialization always uses `experiment_index=0`:**\
In `_lhs_initialize_experiments → _compute_candidate_fim`, `experiment_index=0` is
hardcoded regardless of `_template_mode`.  For user-initialized mode this is wrong if
different experiments have different model structures, though in practice the method is
only reachable via `optimization_method="lhs"` which is tested only for template mode.
See V-4 in `_parallel_initialization.md`.

### 2.2 LHS argument validation

All `lhs_*` arguments are type-checked with `isinstance` only when
`initialization_method == "lhs"`.  Validation is thorough (10 checks), but:

- `lhs_parallel=True` validation uses an f-string bug: `"got {lhs_parallel!r}."` is
  missing the `f` prefix, producing the literal string `{lhs_parallel!r}` in the error
  message.
- `lhs_seed` has no type check — passing e.g. `lhs_seed=1.5` (float) is silently
  accepted and forwarded to `np.random.default_rng(1.5)`, which creates an RNG from a
  float seed (behavior here is numpy-version-dependent).

---

## 3. Model Building Phase

### 3.1 In-place model mutation

`self.model` is a `pyo.ConcreteModel()` created once in `__init__`.  Each call to
`optimize_experiments` adds `param_scenario_blocks` to the **same** `self.model`.  If
called twice:
- `self.model.param_scenario_blocks` is overwritten via Pyomo's `add_component`,
  which raises `DeveloperError: "Attempting to re-use... component"`.
- `self.model.dummy_obj` (added during the square solve, only deactivated—not
  deleted) is still present and would raise the same error on the second call.

**Expected pattern:** either reset `self.model = pyo.ConcreteModel()` at the top of
the method, or `del self.model.param_scenario_blocks` before adding new ones.

### 3.2 `create_doe_model` called for every `(s, k)` pair

```python
for s in range(n_param_scenarios):
    for k in range(n_exp):
        self.create_doe_model(
            model=self.model.param_scenario_blocks[s].exp_blocks[k],
            experiment_index=0 if _template_mode else k,
            _for_multi_experiment=True,
        )
```

For `n_param_scenarios=1` (the only supported case) this is `n_exp` calls.  Each call
builds the full FD scenario block structure.  With `n_exp=10` and 3 FD scenarios per
experiment, this creates 30 Pyomo blocks.  There is no caching or cloning from a
template block — every block is built from scratch.  This is the dominant contributor to
`build_time` for large `n_exp`.

### 3.3 `parameter_scenarios` immediately raises `NotImplementedError`

```python
if parameter_scenarios is None:
    n_param_scenarios = 1
else:
    raise NotImplementedError(...)
```

The parameter `parameter_scenarios` is present in the signature, documented in the
docstring, and referenced in loop comments (`# TODO: Add s_prev = 0 to handle parameter
scenarios`), but passing it always raises.  This creates a misleading API surface.
Either remove the parameter or clearly mark it as experimental/unsupported in the
docstring.

---

## 4. Symmetry Breaking

### 4.1 Loop logs inside the outer loop

```python
for k in range(1, n_exp):
    ...  # add constraint
    self.logger.info(
        f"Added {n_exp - 1} symmetry breaking constraints for scenario {s} ..."
    )
```

The `logger.info` call is **inside** the `for k` loop, so for `n_exp=5` it fires 4
times per scenario with the same message ("Added 4 symmetry breaking constraints...").
Move the log outside the loop:

```python
for k in range(1, n_exp):
    ...  # add constraint

self.logger.info(
    f"Added {n_exp - 1} symmetry breaking constraints for scenario {s} ..."
)
```

### 4.2 Auto-selection of symmetry-breaking variable is order-dependent

When no `sym_break_cons` suffix is provided, the first variable from
`first_exp_block.experiment_inputs` is used.  Pyomo suffix/component ordering is
insertion-order for modern Python (3.7+) dicts, but this is not documented as a contract.
A user who labels inputs in a different order in `get_labeled_model` gets a
different symmetry variable silently.  The auto-selection warning is good, but ideally
the auto-selected variable is also reported as `INFO` (not just `WARNING`) and written
to the diagnostics.

### 4.3 Symmetry-breaking via `sym_break_cons` is not validated

If the user passes multiple variables in `sym_break_cons`, the code warns and uses
only the first.  However, there is no check that the chosen variable is actually an
experiment input (it could be any model variable).  A non-input variable would produce
constraints that neither fix inputs nor break permutation symmetry, silently degrading
performance without any error.

---

## 5. LHS Initialization Wiring

### 5.1 `initialization_time` conflates LHS preprocessing and square solve

```python
lhs_init_diagnostics = None
if initialization_method == "lhs":
    best_initial_points, lhs_init_diagnostics = self._lhs_initialize_experiments(...)
    ...  # set variable values

# [then immediately]
self.solver.solve(self.model, tee=self.tee)   # square solve
initialization_time = sp_timer.toc(msg=None)
```

`initialization_time` is measured from `sp_timer.tic()` (which fires before model
building) up to the end of the square solve.  So `initialization_time` includes:
model build (`build_time`), LHS search, and square solve — all of which are already
separately timed.  This makes `initialization_time` in `results["timing"]` misleading.

**Better:** reset the sub-timer after LHS and before the square solve so that
`initialization_time` measures only the square solve:

```python
sp_timer.toc(msg=None)   # consume and discard LHS time
self.solver.solve(...)
initialization_time = sp_timer.toc(msg=None)
```

### 5.2 LHS values are set via `_get_experiment_input_vars` — applies to all FD scenarios

```python
for var, val in zip(exp_input_vars, best_initial_points[k]):
    var.set_value(val)
```

`_get_experiment_input_vars` returns variables from `fd_scenario_blocks[0]` only.
The remaining FD scenario blocks (1, 2, ..., 2*n_params for central difference) start
from whatever initial values were in the original model.  When the square solve
subsequently fixes all experiment inputs, it **synchronises** all FD blocks from the
fixed values of block 0, so the final result is correct — but a reader unfamiliar with
this would not know why setting only block 0 is sufficient.  Add a comment:

```python
# Setting the base FD scenario block (index 0) is sufficient:
# the square solve propagates this value to all other FD scenario blocks
# through the sensitivity constraints before the main NLP solve.
```

---

## 6. Square Solve (Initialization Phase)

### 6.1 `dummy_obj` left on the model permanently

```python
self.model.dummy_obj = pyo.Objective(expr=0, sense=pyo.minimize)
self.solver.solve(...)
self.model.dummy_obj.deactivate()
```

`dummy_obj` is only deactivated, not deleted.  Consequnces:
1. It clutters `self.model` for downstream inspection.
2. On a second call, `self.model.dummy_obj = ...` tries to assign a component that
   already exists, raising `DeveloperError`.

**Fix:** `del self.model.dummy_obj` after deactivating, or `self.model.del_component("dummy_obj")`.

---

## 7. Scenario-Level Variable Initialization (Pre–Main Solve)

This block (lines ~1000–1112) manually computes:
- `scenario.total_fim[p, q]` = sum of individual FIMs + prior
- L (Cholesky), L_inv, fim_inv, cov_trace for the A-optimality (trace) objective
- determinant, pseudo_trace for the respective objectives

### 7.1 `only_compute_fim_lower` symmetrization applied inconsistently

```python
if self.only_compute_fim_lower:
    total_fim_np = total_fim_np + total_fim_np.T - np.diag(np.diag(total_fim_np))
```

This is correct and appears multiple times (once in the pre-solve initialization and
again in results collection). It should be factored into a helper
(`_symmetrize_lower_tri(mat)`) to avoid divergence if the symmetrization logic ever
needs to change.

---

## 8. Main NLP Solve

### 8.1 `results_message` can be undefined

```python
if type(res.solver.message) is str:
    results_message = res.solver.message
elif type(res.solver.message) is bytes:
    results_message = res.solver.message.decode("utf-8")
self.results["Termination Message"] = results_message
```

If `res.solver.message` is `None` (common for some solvers), neither branch executes
and `results_message` is undefined, causing `UnboundLocalError`.  The same bug appears
in `run_doe`.

**Fix:** add an `else` branch:
```python
else:
    results_message = str(res.solver.message) if res.solver.message is not None else ""
```

---

## 9. Results Collection and Serialization

### 9.1 Dual results structure — flat keys + nested dict

`self.results` contains both a flat "legacy" layer (e.g. `results["FIM"]`,
`results["Build Time"]`, `results["log10 D-opt"]`) and a modern nested structure
(`results["settings"]`, `results["timing"]`, `results["scenarios"]`).  The same data
is stored twice.  Consumers need to know which layer to use.  There is no
`DeprecationWarning` on the flat keys, and no migration guide.

**Recommendation:** choose one representation and deprecate the other;
or at minimum document in the docstring that the nested keys are the intended interface
and the flat keys are preserved for backward compatibility.

### 9.2 `json.dump` can fail on Pyomo/numpy types

```python
with open(results_file, "w") as file:
    json.dump(self.results, file, indent=2)
```

`self.results` contains:
- `res.solver.status` — a Pyomo `SolverStatus` enum (not JSON-serializable)
- `res.solver.termination_condition` — a Pyomo `TerminationCondition` enum
- numpy scalars (`np.float64`) from `np.log10(...)`, `np.linalg.cond(...)` etc.

The `tolist()` calls on the FIM arrays fix those arrays, but the solver status enums and
float-backed numpy scalars will cause `json.dump` to raise `TypeError`.

**Fix:** use a custom encoder:
```python
class _PyomoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "value"):    # Pyomo enum
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

json.dump(self.results, file, indent=2, cls=_PyomoJSONEncoder)
```

### 9.3 FIM metrics computed unconditionally on potentially singular FIM

```python
scenario_results["log10 A-opt"] = np.log10(np.trace(np.linalg.inv(total_fim_np)))
scenario_results["log10 D-opt"] = np.log10(np.linalg.det(total_fim_np))
scenario_results["log10 E-opt"] = np.log10(min(np.linalg.eigvalsh(total_fim_np)))
```

Three failure modes:
1. Singular FIM → `linalg.inv` raises `LinAlgError`.
2. Negative determinant (non-PSD FIM) → `np.log10` of negative → `nan`.
3. Negative minimum eigenvalue → `np.log10` of negative → `nan`.

These should be wrapped in `try/except` with `float("nan")` substitution and a warning,
consistent with the fallback behavior in `_evaluate_objective_for_option`.

### 9.4 `_maximize` set defined twice

```python
# In _evaluate_objective_for_option:
_maximize = {ObjectiveLib.determinant, ObjectiveLib.pseudo_trace}

# In optimize_experiments results collection:
_maximize = {
    ObjectiveLib.determinant,
    ObjectiveLib.pseudo_trace,
    ObjectiveLib.minimum_eigenvalue,
}
```

The two definitions are inconsistent: the results collection includes
`minimum_eigenvalue` in the maximize set but `_evaluate_objective_for_option` does not.
`minimum_eigenvalue` returns `0.0` from the static method regardless.  This discrepancy
should be resolved by centralizing the definition as a class constant:

```python
class DesignOfExperiments:
    _MAXIMIZE_OBJECTIVES = frozenset({
        ObjectiveLib.determinant,
        ObjectiveLib.pseudo_trace,
        ObjectiveLib.minimum_eigenvalue,
    })
```

---

## 10. Control Flow Summary

```
optimize_experiments()
├── validate args
├── build model [O(n_exp) create_doe_model calls]
├── add symmetry-breaking constraints  [O(n_exp) constraints]
├── create_multi_experiment_objective_function
├── if lhs:
│     _lhs_initialize_experiments()   ← documented in _parallel_initialization.md
│     set_value() on exp_blocks[k] inputs
├── square solve (dummy_obj=0, inputs fixed)   ← no termination check
├── initialize L, L_inv, fim_inv, determinant, pseudo_trace from numpy
├── main NLP solve                             ← no termination check
└── collect results → self.results
    ├── legacy flat keys
    └── nested: settings / timing / names / diagnostics / scenarios
```

---

## 11. Issue Summary Table

| ID   | Severity | Location                     | Summary                                                               | Status |
|------|----------|------------------------------|-----------------------------------------------------------------------|--------|
| OE-2 | High     | Results collection           | `results_message` unbound when `res.solver.message` is `None`        | Fixed  |
| OE-3 | High     | Results serialization        | `json.dump` raises on Pyomo enums and `np.float64` scalars            | Fixed  |
| OE-4 | High     | Model reuse / re-entrant use | `self.model` mutated in place; second call raises `DeveloperError`    | Fixed  |
| OE-5 | Medium   | LHS arg validation           | f-string bug in `lhs_parallel` validation error message               | Fixed  |
| OE-6 | Medium   | Symmetry breaking            | `logger.info` fires `n_exp-1` times with identical message per scenario | Open |
| OE-8 | Medium   | FIM metric computation       | `inv`, `det`, `eigvalsh` can raise / return `nan` on bad FIM          | Fixed  |
| OE-9 | Medium   | Timing                       | `initialization_time` includes model build + LHS + square solve       | Fixed  |
| OE-10| Low      | Dummy objective              | `dummy_obj` deactivated but not deleted; breaks on second call        | Fixed  |
| OE-11| Low      | `parameter_scenarios`        | Parameter accepted but immediately raises `NotImplementedError`       | Open   |
| OE-12| Low      | Results structure            | Dual flat+nested results layout; no deprecation path for flat keys    | Open   |
| OE-13| Low      | Symmetry breaking            | `sym_break_cons` variable not validated to be an experiment input     | Fixed  |
| OE-14| Low      | `_maximize` set              | Defined inconsistently in two places; should be a class constant      | Open   |
| OE-15| Low      | `only_compute_fim_lower`     | Symmetrization logic duplicated 2× in the method; extract to helper  | Open   |
| OE-16| Low      | `lhs_seed` validation        | Float seeds accepted silently (only `int` should be valid)            | Fixed  |

---

## 12. Detailed Issue Descriptions

### OE-2 · High · `results_message` unbound when solver message is not str/bytes

`res.solver.message` can be `None` for several solvers (e.g. GLPK, HiGHS).  The
`if/elif` has no `else`, leaving `results_message` unbound and causing
`UnboundLocalError` on the very next line.

**Fix:** add `else: results_message = ""` (or `str(...)`) between the `elif` and the
assignment.

### OE-3 · High · `json.dump` fails on Pyomo enum and numpy scalar types
**Status:** Fixed on current branch.

`res.solver.status` is `SolverStatus.ok` (a Python `str`-backed enum — actually fine),
but `res.solver.termination_condition` is `TerminationCondition.optimal` (also str-backed).
The real risk is `np.float64` from `np.log10(...)` and `np.linalg.cond(...)` — these are
not `float` and `json.JSONEncoder` raises `TypeError` for them.

**Fix:** use a custom `JSONEncoder` that calls `.item()` on numpy scalars and `str()` on
Pyomo enums.

### OE-4 · High · `self.model` mutated in place; calling twice raises `DeveloperError`
**Status:** Fixed on current branch.

`self.model = pyo.ConcreteModel()` in `__init__`, then
`self.model.param_scenario_blocks = pyo.Block(...)` in `optimize_experiments`.  On a
second call, Pyomo raises `DeveloperError: Attempting to re-use a component...`.
`dummy_obj` aggravates this: even if `param_scenario_blocks` is handled, `dummy_obj` is
never deleted.

**Fix:** reset the model at the start of the method:
```python
self.model = pyo.ConcreteModel()
self._built_scenarios = False
```

### OE-5 · Medium · f-string bug in `lhs_parallel` validation

```python
if not isinstance(lhs_parallel, bool):
    raise ValueError(
        "``lhs_parallel`` must be a bool, got {lhs_parallel!r}."  # missing f-prefix
    )
```

The error message will literally say `{lhs_parallel!r}` rather than the actual value.

### OE-6 · Medium · Symmetry-breaking `logger.info` fires inside the loop

See Section 4.1.  Move the `logger.info` outside the `for k in range(1, n_exp)` loop.

### OE-8 · Medium · FIM metric computation can raise or return `nan`

`np.linalg.inv(total_fim_np)` raises `LinAlgError` for singular FIM.
`np.log10(np.linalg.det(...))` returns `nan` for near-zero determinants.
`np.log10(min(np.linalg.eigvalsh(...)))` returns `nan` for negative eigenvalues.

**Fix:** wrap in try/except per metric or check conditioning first:
```python
try:
    scenario_results["log10 A-opt"] = np.log10(np.trace(np.linalg.inv(total_fim_np)))
except np.linalg.LinAlgError:
    scenario_results["log10 A-opt"] = float("nan")
```

### OE-9 · Medium · `initialization_time` measures too much
**Status:** Fixed on current branch.

`sp_timer.tic()` fires before model building, and `initialization_time = sp_timer.toc()`
fires after the square solve.  So `initialization_time ≥ build_time + lhs_time + square_solve_time`.
The `timing` dict exposes `initialization_s` separately from `build_s` and `solve_s`,
implying it should only cover the square solve.

**Fix:** reset the sub-timer between the LHS step and the square solve call.

### OE-10 · Low · `dummy_obj` deactivated but not deleted
**Status:** Fixed on current branch.

See Section 6.1.  Add `self.model.del_component("dummy_obj")` after deactivating.

### OE-11 · Low · `parameter_scenarios` accepted but always raises

Either remove the parameter or clarify it is not yet supported in the docstring with
`.. versionadded:: (future)`.

### OE-12 · Low · Dual flat+nested results structure lacks deprecation guidance

Flat keys like `results["Build Time"]`, `results["FIM"]` are the historic interface.
The nested `results["timing"]`, `results["scenarios"]` are the modern interface.
Both exist simultaneously with no guidance.  Add a deprecation warning when accessing
flat keys, or document clearly which interface is primary.

### OE-13 · Low · `sym_break_cons` variable not validated
**Status:** Fixed on current branch.

If a user puts a non-experiment-input variable in `sym_break_cons`, the constraint
`var_prev <= var_curr` is silently added but doesn't constrain the experiment design.
Add a check:

```python
if sym_break_var not in set(first_exp_block.experiment_inputs.keys()):
    self.logger.warning(
        f"sym_break_cons variable '{sym_break_var.local_name}' is not in "
        "experiment_inputs. Symmetry breaking constraint may have no effect."
    )
```

### OE-14 · Low · `_maximize` defined inconsistently in two places

See Section 9.4.  Centralise as a class-level constant.

### OE-15 · Low · Lower-triangle symmetrization duplicated twice

```python
# appears in pre-solve initialization AND results collection:
total_fim_np = total_fim_np + total_fim_np.T - np.diag(np.diag(total_fim_np))
```

Extract to a static helper: `_symmetrize_lower_tri(mat: np.ndarray) -> np.ndarray`.

### OE-16 · Low · `lhs_seed` accepts float silently
**Status:** Fixed on current branch.

`numpy.random.default_rng(1.5)` does not raise but creates an RNG with
implementation-defined behavior.  Add `isinstance(lhs_seed, int)` check alongside
the `is not None` guard.

---

## 13. Missing Test Coverage

| Gap | Description |
|-----|-------------|
| Second `optimize_experiments` call on same DoE object | Covered (regression test added for OE-4 re-entrant behavior) |
| `results_file` output with real solver run | Covered (regression test writes + loads JSON results) |
| `parameter_scenarios != None` | Should verify `NotImplementedError` is raised |
| `sym_break_cons` with a non-experiment-input variable | Covered (regression test now expects validation error) |
| `n_exp=1` with `initialization_method="lhs"` | Edge case: `C(N,1)=N`; trivially best is highest-obj single candidate |
| User-initialized mode + `initialization_method="lhs"` | Currently each experiment should sample from its own model, but `experiment_index=0` is hardcoded (V-4) |

---

## 14. Related Documents

- [`_parallel_initialization.md`](_parallel_initialization.md) — full review history
  of `_lhs_initialize_experiments` (Rounds 1–5).
- [`doe.py`](doe.py) — source; primary changes needed at lines
  ~699–702 (file validation), ~731–783 (lhs arg validation), ~831–907 (symmetry
  breaking), ~924–948 (LHS wiring), ~962–984 (square solve), ~1000–1112 (pre-solve init),
  ~1120 (main solve), ~1130–1243 (results collection), ~1395 (JSON dump).
