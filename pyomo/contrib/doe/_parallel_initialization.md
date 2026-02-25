# Parallel Initialization Design Notes

This document captures the current agreed design for parallelizing the LHS initialization path used by `DesignOfExperiments.optimize_experiments()`.

## Scope
- Target code path: LHS initialization for multi-experiment design selection.
- Current objective: keep exact combination search (no greedy/beam yet).
- Output requirement: only best `n_exp` initialization points; no need to store all evaluated combinations.

## Problem Summary
Given candidate design points from LHS/cartesian-product sampling:
- Compute candidate FIMs `F_i` for `i = 1..N`.
- Choose a subset of size `k = n_exp` maximizing/minimizing objective on:
  - `F_total = F0 + sum(F_i over selected indices)`
  - where `F0` is prior FIM.

Bottlenecks:
1. Candidate FIM computation (`N` expensive solver calls).
2. Exact combination scoring (`C(N, k)` can be very large).

## Final Agreed Algorithm (Exact + Parallel)

### Stage A: Parallel candidate FIM computation
1. Build candidate point list.
2. Evaluate `F_i` for each candidate in parallel (thread workers).
3. Store ordered `candidate_fims`.

Notes:
- Workers compute independent tasks; no shared solver/model state.
- If parallel is disabled, fallback is serial by construction.

### Stage B: Exact combination scoring (serial or parallel)
Let `M = C(N, k)`.

Decision rule:
- If `M` is small: serial exact scoring.
- If `M` is large and combo-parallel enabled: parallel chunked scoring.

#### Serial exact scoring
- Iterate combinations lazily.
- Score each combo exactly.
- Keep global best `(best_obj, best_combo)`.

#### Parallel exact scoring (map-reduce)
- Stream combinations lazily in chunks.
- Each worker scores one chunk and returns only local best `(obj, combo)`.
- Parent reduces worker-local bests to global best.

Important correctness point:
- For global best-1, worker-local best-1 is sufficient.
- Keeping top-`k` per worker is only needed if global top-`k` output is required.

### Optional fast path for `k == 2`
- For pair selection, score `(i, j)` directly via:
  - `F_total = F0 + F_i + F_j`
- Same exact objective, lower overhead.

## Memory Strategy
- Store only:
  - candidate points
  - `candidate_fims`
  - current best tuple(s)
- Do not materialize all combinations.
- Stream combination iterator and process chunks.

## Proposed Runtime Controls
These are candidate API/config parameters for LHS initialization:
- `lhs_parallel` (bool)
- `lhs_n_workers` (int or None)
- `lhs_combo_parallel` (bool)
- `lhs_combo_chunk_size` (int)
- `lhs_combo_parallel_threshold` (int)
- `lhs_max_wall_clock_time` (seconds, optional)

Timeout behavior (proposed):
- Return best-so-far initialization for robustness in initialization context.
- Record timeout flag/message in diagnostics.

## Why this path first
- Preserves exactness (same best combo as exhaustive serial, barring timeout).
- Attacks both major bottlenecks.
- Bounded memory via streaming.
- Lower risk than introducing approximate search immediately.

## Deferred / Future Work
1. Approximate modes for very large search spaces:
   - greedy forward
   - beam search
2. MPI backend option (possibly via `parmest` MPI utilities) after process-based path stabilizes.
3. Top-K combinations output mode (if required by users).

## Validation Plan
1. Unit tests: serial vs parallel exact equivalence on small deterministic cases.
2. Chunk boundary tests (non-divisible chunk counts).
3. `k==2` path equivalence with generic path.
4. Timeout behavior tests (best-so-far contract).
5. Performance smoke tests (no strict walltime assertions in CI).

## Update Policy
This file is a living design record.
- Any algorithm or API decision changes should be reflected here.
- Keep implementation status and tradeoff decisions synchronized with code reviews.

## Implementation Status

### Milestone 1: API and wiring
- [x] Add LHS parallel control arguments (parallel/workers/chunk/threshold/timeout)
- [x] Validate new arguments and default behavior
- [x] Keep backward compatibility for existing calls

### Milestone 2: Stage A parallel FIM evaluation
- [x] Implement parallel map for candidate FIM evaluations
- [x] Preserve deterministic candidate ordering in outputs
- [x] Keep serial fallback path when `lhs_parallel=False` or single worker

### Milestone 3: Stage B exact combination scoring
- [x] Keep current serial exact scorer as baseline
- [x] Add chunked parallel map-reduce scorer for combinations
- [x] Implement reducer for global best-1 `(obj, combo)`
- [x] Add optional `k == 2` fast path (exact)

### Milestone 4: Timeout and diagnostics
- [x] Implement `lhs_max_wall_clock_time` handling
- [x] Define behavior on timeout (best-so-far return)
- [x] Record diagnostics fields (mode, workers, timings, timeout/warnings)

### Milestone 5: Tests
- [x] Serial vs parallel exact-equivalence tests (small deterministic case)
- [x] Chunk boundary tests (non-divisible chunk sizes)
- [x] `k == 2` fast path equivalence tests
- [x] Timeout behavior tests
- [ ] Performance smoke tests (non-strict)

### Milestone 6: Docs and cleanup
- [ ] Update user/developer docs for new initialization options
- [ ] Add changelog notes
- [ ] Remove dead code paths and finalize logging messages

### Current State
- Design finalized in this file: **Yes**
- Code implementation started: **Yes (Milestones 1, 2, 3, 4)**
- Test implementation started: **Yes (Milestone 5 in progress)**

---

## Parallel Initialization Feedback

### Round 1 — Issues raised and resolution status

| ID  | Severity | Summary                                                      | Status      |
|-----|----------|--------------------------------------------------------------|-------------|
| C-1 | Critical | `ThreadPool` for CPU-bound solvers; GIL prevents Stage A speedup | **Open (deferred)** — see below |
| C-2 | Critical | Full `DesignOfExperiments` reconstructed per candidate       | **Fixed** — `threading.local()` worker reuse |
| C-3 | Critical | `fut.cancel()` no-op on in-flight threads                    | **Substantially fixed** — `deadline_ts` passed into `_score_chunk`; cooperative check at top of inner loop; `fut.cancel()` now only applies to not-yet-started futures (correct semantics) |
| H-1 | High     | Pre-timeout fallback returned unscored default combo         | **Fixed** — `best_combo = None` init; fallback only triggered when `None` |
| H-2 | High     | Timeout check was post-score                                 | **Fixed** — deadline checked at TOP of serial loop body |
| M-1 | Medium   | `_evaluate_objective_from_fim` called across threads via `self` | **Fixed** — extracted as `_evaluate_objective_for_option` `@staticmethod` |
| M-2 | Medium   | `lhs_n_workers` mutated unconditionally                      | **Fixed** — `resolved_workers` local variable |
| M-3 | Medium   | LHS diagnostics missing from `self.results`                  | **Fixed** — `lhs_init_diagnostics` dict wired into `results["diagnostics"]` |
| L-1 | Low      | `k = n_exp` alias shadowing in `_score_chunk`                | **Fixed** — `n_exp == 2` used inline |
| L-2 | Low      | `toc()` cumulative totals, not delta timings                 | **Fixed** — `time.perf_counter()` checkpoints per stage |
| T-1 | Medium   | Worker DoE construction path untested                        | **Fixed** — `test_lhs_parallel_fim_eval_real_path_smoke` added |
| T-2 | Medium   | Parallel combo timeout had no test                           | **Fixed** — `test_lhs_combo_scoring_parallel_timeout_returns_best_so_far` added |
| T-3 | Low      | `itertools.combinations` patch target fragile to import hoisting | **Fixed** — `from itertools import combinations as _combinations` hoisted to module level; patch target is now `"pyomo.contrib.doe.doe._combinations"` |

---

### Round 2 — New findings after the above fixes

| ID  | Severity | Summary                                                     | Status |
|-----|----------|-------------------------------------------------------------|--------|
| N-1 | Medium   | `_score_chunk` read `self.objective_option` implicitly      | **Fixed** — snapshot `_obj_option` before threaded scoring |
| N-2 | Low      | `_lhs_initialize_experiments` had polymorphic return type   | **Fixed** — always returns `(best_initial_points, diagnostics)` |
| N-3 | Medium   | real-path parallel smoke test too thin (`lhs_n_samples=2`)  | **Fixed** — raised to `lhs_n_samples=3` |
| N-4 | Low      | diagnostics test only checked key presence                  | **Fixed** — now validates expected values |
| N-5 | Low      | solver fallback in `_make_worker_solver` was silent         | **Fixed** — added debug fallback logging |

---

### Round 3 — New findings after Round 2 fixes

| ID  | Severity | Summary                                                            | Status |
|-----|----------|--------------------------------------------------------------------|--------|
| R-1 | Low      | `update_model_from_suffix` imported inside method body on every call | **Fixed** — hoisted to module-level import (line 66) |
| R-2 | Low      | List-as-mutable-closure `[False]` used instead of `nonlocal`      | **Fixed** — replaced with `nonlocal` boolean |
| R-3 | Medium   | Worker constructor exceptions unhandled in parallel FIM path       | **Fixed** — `try/except` wraps full `_compute_candidate_fim` body; `n_params` resolved before closure |
| R-4 | Low      | `best_obj` not recorded in `lhs_diagnostics`                      | **Fixed** — `"best_obj": best_obj` added to diagnostics dict |
| R-5 | Low      | Timeout tests did not verify `timed_out` flag in diagnostics      | **Fixed** — both timeout tests now assert `diag["timed_out"]` |

---

### Round 4 — New findings after Round 3 fixes

| ID  | Severity | Summary                                                                  | Status |
|-----|----------|--------------------------------------------------------------------------|--------|
| S-1 | Low      | `LatinHypercube` still imported inside method body                       | **Fixed** — hoisted to module-level conditional import |
| S-2 | Low      | `n_params = self.prior_FIM.shape[0]` has implicit pre-condition          | **Fixed** — resolved from model block (`fd_scenario_blocks[0].unknown_parameters`) instead of `prior_FIM` |
| S-3 | Low      | `@staticmethod` called via `self.` in `_score_chunk` — misleading style | **Fixed** — uses local `_score_obj = DesignOfExperiments._evaluate_objective_for_option` |
| S-4 | Low      | Diagnostics test only checks `best_obj` key presence, not type/value    | **Fixed** — test now asserts type/finite/positive value |
| S-5 | Medium   | Generic `n_exp >= 3` FIM accumulation path in `_score_chunk` is untested | **Fixed** — added `n_exp=3` serial/parallel equivalence test |

#### S-1 · Low · `LatinHypercube` still imported inside the method body

```python
# inside _lhs_initialize_experiments:
from scipy.stats.qmc import LatinHypercube
```

`update_model_from_suffix` was correctly hoisted to module level (R-1), but `LatinHypercube`
still re-executes its import on every call.  The `scipy_available` guard that wraps the
module already prevents `ImportError` at import time.
**Fix:** Add to the top-of-file conditional import block alongside the other scipy
imports.

#### S-2 · Low · `n_params = int(self.prior_FIM.shape[0])` has implicit pre-condition

```python
n_params = int(self.prior_FIM.shape[0])
```

This line (added to fix R-3) will raise `AttributeError` if `self.prior_FIM` is `None`
(its default value from the constructor).  In the normal `optimize_experiments` call flow
this is safe because `create_doe_model` sets `self.prior_FIM` before
`_lhs_initialize_experiments` is called.  However, a direct call to
`_lhs_initialize_experiments` (as done in all helper tests via
`_build_template_model_for_multi_experiment`) skips this setup, relying on the default
`prior_FIM=None` being set to zeros somewhere else first.

**Fix:** Add a guard at the top of `_lhs_initialize_experiments`:
```python
if self.prior_FIM is None:
    raise RuntimeError(
        "_lhs_initialize_experiments requires prior_FIM to be set. "
        "Call via optimize_experiments or set prior_FIM explicitly."
    )
```
Or, more robustly, resolve `n_params` from the experiment model itself rather than from
`prior_FIM`.

#### S-3 · Low · `@staticmethod` called via `self.` inside `_score_chunk`

```python
obj_val = self._evaluate_objective_for_option(fim_total, _obj_option)
```

`_evaluate_objective_for_option` is `@staticmethod` — it doesn't receive `self`.  Calling
it via `self.` works correctly in Python but reads as an instance method call and hides
the thread-safety guarantee.  In a code review or future refactor, a developer may
mistakenly assume `self` is being read inside that method.
**Fix:** Replace with an explicit class-level or module-level call:
```python
obj_val = DesignOfExperiments._evaluate_objective_for_option(fim_total, _obj_option)
```
Or assign the static method to a local before the closure is defined:
```python
_score_obj = DesignOfExperiments._evaluate_objective_for_option
# then inside _score_chunk:
obj_val = _score_obj(fim_total, _obj_option)
```

#### S-4 · Low · `best_obj` in diagnostics test only checks key presence

```python
self.assertIn("best_obj", lhs_diag)
```

After R-4 added `best_obj` to the diagnostics, the diagnostics test was updated to check
presence but not value or type.  A regression that stores `None` or `np.inf` mistakenly
would not be caught.
**Fix:**
```python
self.assertIsInstance(lhs_diag["best_obj"], float)
self.assertGreater(lhs_diag["best_obj"], 0.0)  # pseudo_trace of a positive FIM
```

#### S-5 · Medium · Generic `n_exp >= 3` path in `_score_chunk` has no test coverage

Every test in `TestOptimizeExperimentsAlgorithm` uses `n_exp=2`, which always hits the
fast path:
```python
if n_exp == 2:
    fim_total = prior + candidate_fims[i] + candidate_fims[j]
```

The `else` branch (iterative `fim_total = prior.copy(); for idx in combo: fim_total += ...`)
is exercised by **no test at all** — neither serial, parallel, nor timeout.  This code
path handles all multi-experiment designs beyond pairs and is the default for most
practical use cases with `n_exp >= 3`.

**Fix:** Add a test with `n_exp=3` (e.g. `lhs_n_samples=2` → 8 candidates,
`C(8, 3) = 56` combinations):
```python
def test_lhs_combo_scoring_n_exp_3_serial(self):
    doe = self._make_template_doe("pseudo_trace")
    self._build_template_model_for_multi_experiment(doe, n_exp=3)
    ...
    points, _ = doe._lhs_initialize_experiments(
        lhs_n_samples=2, lhs_seed=42, n_exp=3
    )
    self.assertEqual(len(points), 3)
```

---

### Round 5 — New findings after Round 4 fixes

| ID  | Severity | Summary                                                                                       | Status |
|-----|----------|-----------------------------------------------------------------------------------------------|--------|
| V-1 | Medium   | `best_obj` in diagnostics is sentinel `±inf` when fallback combo is used after early timeout  | **Fixed** — fallback combo is now scored and `best_obj` updated |
| V-2 | Medium   | Worker thread constructor failure is silent: retried every task, no consolidated warning       | **Fixed** — per-thread construction-failure sentinel + consolidated error log |
| V-3 | Low      | `fd_scenario_blocks[0]` access lacks `hasattr` guard — inconsistent with `n_scenarios_per_candidate` block above | **Fixed** — explicit precondition guard with clear `RuntimeError` |
| V-4 | Low      | `experiment_index=0` hardcoded in `_compute_candidate_fim` — undocumented restriction         | **Fixed** — rationale comment added in worker evaluation path |
| V-5 | Low      | `n_exp=3` test only proves serial == parallel; no oracle check guards against a shared bug    | **Fixed** — added oracle-style deterministic `n_exp=3` test |

#### V-1 · Medium · `best_obj` in diagnostics is uninitialized sentinel when fallback combo is used

```python
best_obj = -np.inf if is_maximize else np.inf
best_combo = None
...
# (all combos timed out before any was scored)
if best_combo is None:
    best_combo = tuple(range(n_exp))   # fallback
# best_obj is never updated → stores ±inf
lhs_diagnostics = { ..., "best_obj": best_obj }   # ±inf
```

When `lhs_max_wall_clock_time` is extremely tight, the deadline can fire before a single
combination is scored.  `best_combo` falls back to `tuple(range(n_exp))`, but `best_obj`
remains the sentinel (`-inf` for maximize, `+inf` for minimize) and is persisted into
`lhs_diagnostics["best_obj"]`.  Any downstream code reading that value to judge
initialization quality (e.g. a convergence criterion or a user-facing log) will
see a meaningless sentinel rather than the actual objective of the fallback design.

**Fix:** After the fallback, recompute `best_obj` for the chosen combo:
```python
if best_combo is None:
    best_combo = tuple(range(n_exp))
    self.logger.warning(
        "LHS combination scoring ended before any combination was scored. "
        "Falling back to the first n_exp candidate points."
    )
    # score the fallback so diagnostics are meaningful
    if n_exp == 2:
        i, j = best_combo
        fim_fb = prior + candidate_fims[i] + candidate_fims[j]
    else:
        fim_fb = prior.copy()
        for idx in best_combo:
            fim_fb = fim_fb + candidate_fims[idx]
    best_obj = float(_score_obj(fim_fb, _obj_option))
```
Alternatively, store `best_obj = None` when no combo was scored and handle `None`
explicitly in any code that reads the diagnostics.

#### V-2 · Medium · Worker thread constructor failure degrades silently per-task with no consolidated warning

In `_compute_candidate_fim`, if `DesignOfExperiments(...)` raises an exception, the call
to `thread_state.doe = worker_doe` never executes.  The next task dispatched to the same
thread finds `thread_state.doe is None`, tries to construct again, fails again, and
returns another zero FIM — and so on for every candidate assigned to that thread.  Each
failure emits an individual per-candidate `WARNING`, but there is no consolidated
message like:

> *"LHS: thread-0 worker initialization failed; all N candidates on this thread have
> zero FIM — parallel results may be unreliable."*

In a run with `resolved_workers=4` and `n_candidates=100`, one broken thread silently
pollutes ~25 candidates.  A user who only checks the final best-obj value will not
otice unless they scan logs carefully.

**Fix:** Track construction failure in `thread_state`:
```python
if worker_doe is None:
    if getattr(thread_state, "construction_failed", False):
        # Already tried and failed; skip logging noise
        return idx, np.zeros((n_params, n_params))
    try:
        worker_solver = _make_worker_solver()
        worker_doe = DesignOfExperiments(...)
        thread_state.doe = worker_doe
    except Exception as exc:
        thread_state.construction_failed = True
        self.logger.error(
            f"LHS: worker DoE construction failed (thread {threading.current_thread().name}): "
            f"{exc}. All candidates on this thread will use zero FIM."
        )
        return idx, np.zeros((n_params, n_params))
```

#### V-3 · Low · `fd_scenario_blocks` access lacks `hasattr` guard — inconsistent with nearby code

```python
# Lines ~1591-1594: guarded access
if hasattr(first_exp_block, "fd_scenario_blocks"):
    n_scenarios_per_candidate = len(list(first_exp_block.fd_scenario_blocks))
else:
    n_scenarios_per_candidate = 1

# Line ~1614: unguarded access that raises AttributeError if fd_scenario_blocks absent
n_params = len(first_exp_block.fd_scenario_blocks[0].unknown_parameters)
```

The code already recognises that `fd_scenario_blocks` may not always be present on the
experiment block (the `hasattr` guard for `n_scenarios_per_candidate`).  Three lines later,
the same attribute is accessed unconditionally.  If the attribute is absent the `AttributeError`
will surface as an unhelpful traceback instead of a clear message about preconditions.

**Fix:** Add a guard consistent with the existing pattern:
```python
if not hasattr(first_exp_block, "fd_scenario_blocks") or not first_exp_block.fd_scenario_blocks:
    raise RuntimeError(
        "_lhs_initialize_experiments requires the experiment model to be built "
        "with finite-difference scenario blocks (sequential FIM method). "
        "Ensure the model is created via optimize_experiments with method='sequential'."
    )
n_params = len(first_exp_block.fd_scenario_blocks[0].unknown_parameters)
```

#### V-4 · Low · `experiment_index=0` is hardcoded in `_compute_candidate_fim` without documentation

```python
fim = worker_doe._compute_fim_at_point_no_prior(
    experiment_index=0, input_values=list(pt)
)
```

LHS initialization evaluates design points for a **single** canonical experiment type,
always using `experiment_list[0]`.  This is correct for the current use-case (all
experiments share the same model structure with different input values), but the
assumption is not recorded anywhere in the code or the method docstring.  A developer
who adds multi-type experiment list support in the future could introduce a silent
bug by calling `_lhs_initialize_experiments` without realising only the first
experiment type is evaluated.

**Fix:** Add a comment and/or a docstring note on `_lhs_initialize_experiments`:
```python
# All LHS candidates are evaluated using experiment_list[0] as the canonical
# experiment template.  This is valid when all experiments share the same
# model/parameter structure (the standard multi-experiment DoE case).
fim = worker_doe._compute_fim_at_point_no_prior(
    experiment_index=0, input_values=list(pt)
)
```

#### V-5 · Low · `n_exp=3` test proves only serial == parallel, not correctness

```python
serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
self.assertEqual(serial_norm, parallel_norm)   # mutual agreement only
```

The new test (`test_lhs_combo_scoring_n_exp_3_parallel_matches_serial`) verifies that
serial and parallel combo scoring agree for `n_exp=3`.  This is valuable for regression
detection, but a shared logical bug in both paths (e.g. an off-by-one in the combo
iterator, or a wrong accumulation of FIM slices) would be undetected since both paths
would still agree with each other.

**Fix:** For `n_exp=3` with a small deterministic synthetic case (e.g. 3 candidates,
`C(3,3)=1` combination), verify the expected best combo explicitly:
```python
def test_lhs_combo_scoring_n_exp_3_oracle(self):
    doe = self._make_template_doe("pseudo_trace")
    self._build_template_model_for_multi_experiment(doe, n_exp=3)
    # 3 candidates → only one possible triple (0,1,2)
    fake_fims = [
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        np.array([[3.0, 0.0], [0.0, 4.0]]),
        np.array([[5.0, 0.0], [0.0, 6.0]]),
    ]
    with patch.object(doe, "_compute_fim_at_point_no_prior",
                      side_effect=lambda ei, iv: fake_fims[int(float(iv[0]))]):
        points, diag = doe._lhs_initialize_experiments(
            lhs_n_samples=3,
            lhs_seed=0,
            n_exp=3,
            lhs_combo_parallel=False,
        )
    # prior is 2x2 zeros; FIM_total = F0+F1+F2 → trace = (1+3+5)+(2+4+6) = 21
    self.assertAlmostEqual(diag["best_obj"], 21.0, places=6)
    self.assertEqual(len(points), 3)
```

---

### Open Item — Stage A `ThreadPoolExecutor` vs `ProcessPoolExecutor`

The current deferral rationale is: *"Moving to a process backend is deferred due to higher
cross-platform complexity (pickling experiment objects, solver/process lifecycle)."*

**Assessment after the `threading.local()` fix (C-2):**

The fix in C-2 means the per-task overhead of a full `DesignOfExperiments` constructor is
gone — each thread now creates one worker DoE on first use and reuses it.  The practical
question is therefore: **how much real parallelism do threads give for this workload?**

The workload per candidate is:
1. `experiment.get_labeled_model().clone()` — pure Python, GIL-bound, ~10 ms range.
2. `update_model_from_suffix` — pure Python, GIL-bound.
3. `solver.solve(model)` — launches IPOPT as a `subprocess.Popen`.  The Python interpreter
   *releases the GIL* while blocked on subprocess I/O.  This is typically the dominant cost
   (seconds per solve).

Because step 3 dominates and releases the GIL, **threads already provide meaningful
parallelism for solver-heavy experiments**.  The remaining GIL-bound steps (1–2) serialize,
but they are small relative to the solver call.  The deferred rationale is therefore
**valid and reasonable for the current use case**.

A `ProcessPoolExecutor` migration would be worthwhile if:
- Pyomo model construction becomes a bottleneck (e.g. very large models with many constraints).
- The experiment class is not automatically picklable (common for objects with Pyomo components),
  which would require explicit `__getstate__`/`__setstate__` implementation.

**Recommended path for deferral:**
1. Measure the fraction of time in steps 1–2 vs step 3 on a representative experiment.
   If > 10 % of wall-clock is in Python model setup with `lhs_n_workers > 1`, re-evaluate
   the process switch.
2. When switching, use the `ProcessPoolExecutor` `initializer=` pattern:
   pass only the picklable experiment *factory* (callable + arguments) and reconstruct the
   model inside each worker process, rather than pickling a live Pyomo `ConcreteModel`.
3. Consider `multiprocessing.get_context("spawn")` explicitly for cross-platform safety
   (avoids `fork`-unsafe patterns with IPOPT file handles on macOS/Linux).
