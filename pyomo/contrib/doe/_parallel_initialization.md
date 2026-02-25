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
- [ ] Record diagnostics fields (mode, workers, timings, timeout/warnings)

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
- Code implementation started: **Yes (Milestones 1, 2, 3; partial 4)**
- Test implementation started: **Yes (Milestone 5 in progress)**
