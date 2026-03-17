# DesignOfExperiments `optimize_experiments()` Proposed Interface

This is a brief documentation explaining the changes. This document will not get merged.

## Why This Change

The `optimize_experiments()` path has grown from a single-experiment workflow into a 
multi-experiment optimization interface with optional initialization strategies and 
richer result payloads. This document captures the interface contract, operating modes, 
and optimization model at a high level.

## API Summary

### Constructor

```python
DesignOfExperiments(
    experiment=...,   # single experiment object OR list of experiment objects
    ...
)
```

- `experiment` now accepts either:
  - one experiment object (template mode input), or
  - a list of experiment objects (user-initialized mode input).
- Internally, the implementation normalizes this to `self.experiment_list`.

### Multi-Experiment Solve Entry Point

```python
optimize_experiments(
    results_file=None,
    n_exp: int = None,
    init_method: InitializationMethod = None,
    init_n_samples: int = 5,
    init_seed: int = None,
    init_parallel: bool = False,
    init_combo_parallel: bool = False,
    init_n_workers: int = None,
    init_combo_chunk_size: int = 5000,
    init_combo_parallel_threshold: int = 20000,
    init_max_wall_clock_time: float = None,
    init_solver=None,           
)
```

### Short description of arguments

- `results_file`: Optional path for writing JSON results.
- `n_exp`: Number of experiments to optimize in template mode.
- `init_method`: Initialization strategy (`None` or `"lhs"`).
- `init_n_samples`: Number of LHS samples per experiment-input dimension.
- `init_seed`: Random seed used by LHS initialization.
- `init_parallel`: Enables parallel candidate-point FIM evaluation.
- `init_combo_parallel`: Enables parallel candidate-combination scoring.
- `init_n_workers`: Worker count for LHS parallel evaluation/scoring paths.
- `init_combo_chunk_size`: Number of combinations handled per worker task.
- `init_combo_parallel_threshold`: Minimum combinations required before using combo parallelism.
- `init_max_wall_clock_time`: Optional LHS time budget (seconds); returns best-so-far if exceeded.
- `init_solver`: Optional solver used only during initialization phases (LHS and square init solve).

## Operating Modes

### 1) Template Mode

- Condition: `len(experiment) == 1`
- `n_exp` may be provided to choose how many experiments to optimize simultaneously.
- If `n_exp` is omitted, default is `1`.

### 2) User-Initialized Mode

- Condition: `len(experiment) > 1`
- Number of experiments is fixed by the list length.
- `n_exp` must not be provided.

## Initialization Behavior

### No Special Initialization (`init_method=None`)

- Uses current experiment design values from the model labels directly.

### LHS Initialization (`init_method="lhs"`)

- Currently supported in template mode.
- Requires explicit lower and upper bounds for all experiment inputs.
- Generates candidate points using per-dimension 1-D LHS, then Cartesian product.
- Scores combinations of candidate points using objective-specific FIM metrics.
- Selects best-scoring set of initial points for the nonlinear solve.

### `init_solver` (new)

- If provided, `init_solver` is used for:
  - initialization-phase solves (including LHS candidate FIM evaluation path),
  - the square initialization solve before final optimization.
- Final optimization solve still uses the main DoE solver (`self.solver`).
- If `init_solver` is `None`, initialization also uses `self.solver`.

## Optimization Formulation

The proposed multi-experiment interface follows a simultaneous design formulation.

### General Form

Let:

- $E = \{1, 2, ..., N_{exp}\}$ be the experiment index set,
- $phi_k$ be the design variables for experiment `k`,
- $M_0$ be the prior FIM,
- $M_k(\hat{\theta}, \phi_k)$ be the FIM contribution from experiment `k`,
- $\Psi(M)$ be the chosen FIM metric (D-, A-, pseudo-A-, etc.).

Then:

```math
\max_{\phi_1,\ldots,\phi_{N_{exp}}} \Psi(\mathbf{M})
```

subject to:

```math
\mathbf{M} = \sum_{k=1}^{N_{exp}} \mathbf{M}_k(\hat{\theta}, \phi_k) + \mathbf{M}_0
```

```math
\mathbf{M}_k = \mathbf{Q}_k^\top \Sigma_{\bar{y},k}^{-1} \mathbf{Q}_k, \quad \forall k \in E
```

```math
\mathbf{m}(\bar{x}_k, \hat{\bar{y}}_k, \phi_k, \hat{\theta}, t) = 0, \quad \forall k \in E
```

```math
\mathbf{g}(\bar{x}_k, \hat{\bar{y}}_k, \phi_k, \hat{\theta}, t) \le 0, \quad \forall k \in E
```

where $\mathbf{Q}_k$ is the sensitivity matrix for experiment `k`.

### Symmetry-Breaking Constraint

To avoid permutation-equivalent solutions in simultaneous design:

```math
\varphi_{\text{primary},1} \le \varphi_{\text{primary},2} \le \cdots \le \varphi_{\text{primary},N_{exp}}
```

This is implemented in `optimize_experiments()` by using a user-marked primary design 
variable passed in a `Pyomo.Suffix` (or a default selection (first variable from 
experiment_inputs Suffix) with warning if not marked).

### Current Implementation Specialization

The current implementation corresponds to the single-scenario case (`N_s = 1`) with:

```math
\mathbf{M}_{\text{total}} = \mathbf{M}_0 + \sum_{k=1}^{N_{exp}} \mathbf{M}_k
```
Future implementation will handle parametric uncertainty with $N_s >1$

Objective handling in code:

- Determinant and pseudo-trace: solved in maximization form (with monotonic transforms used in the NLP objective expressions).
- Trace: solved in minimization form via covariance/FIM-inverse representation.
- Zero objective: feasibility/debug mode.

Cholesky-based constraints and variables are used in supported objective paths to stabilize determinant/trace formulations.

## Result Payload Highlights

The solver output includes both legacy fields and structured fields:

- `run_info` (API name, solver status/termination info)
- `settings` (objective, finite-difference, initialization config, modeling mode)
- `timing` (build/initialization/solve/total timing)
- `names` (design/output/parameter/error labels)
- `diagnostics` (symmetry and LHS diagnostics)
- `scenarios` (scenario-level objective metrics, total FIM, per-experiment details)

Notably, initialization settings now include:

- `settings["initialization"]["solver_name"]`

to make it explicit which solver was used during initialization.

