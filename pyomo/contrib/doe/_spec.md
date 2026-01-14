# Specification: Simultaneous Design of Experiments (Deterministic)

## 1. Overview
We are extending `Pyomo.DoE` to support **simultaneous design of multiple experiments** ($N_{exp} \ge 1$). 
**Constraint:** Currently restricted to deterministic parameters ($N_s=1$), but the Pyomo Block structure **must** utilize a Scenario-Experiment hierarchy (`model.scen_block[s].exp_block[k]`) to allow for seamless future extension to robust optimization.

**Target File:** `pyomo/contrib/doe/doe.py`
**Key Method:** `DesignOfExperiments.optimize_experiments()`

---

## 2. Mathematical Formulation (Deterministic Implementation)
We maximize the Determinant of the Fisher Information Matrix (D-optimality) for a single nominal parameter scenario.

### Optimization Problem
$$
\max_{\boldsymbol{\phi}} \quad \det\left( \sum_{k=1}^{N_{exp}} \mathbf{M}_{k} + \mathbf{M}_{prior} \right)
$$

### Constraints
1.  **Symmetry Breaking:** To prevent permutations of identical experiments:
    $$\varphi_{k, \text{primary}} \le \varphi_{k+1, \text{primary}} \quad \forall k \in \{1, \dots, N_{exp}-1\}$$

---

## 3. Implementation Logic

### Method: `optimize_experiments`
**Location:** `pyomo/contrib/doe/doe.py` inside class `DesignOfExperiments`

#### Signature
```python
def optimize_experiments(
    self, 
    n_exp: int, 
    # parameter_scenarios argument reserved for future use
    objective_option: str = "determinant", 
    solver: Optional[str] = 'ipopt'
):
```

#### Algorithm Steps

1.  **Scenario Setup (Fixed for deterministic):**
    * Set `Ns = 1` (Number of scenarios).
    * Set `weights = [1.0]`.
    * *Future-proofing:* Maintain `Ns` and `weights` as iterable structures (lists/arrays) rather than scalars to ensure the loop logic remains valid when uncertainty is added later.

2.  **Block Construction (Hierarchy Setup):**
    * Create a top-level block `self.model.scen_block` indexed by `s` in `{0, ..., Ns-1}`.
    * Inside each `scen_block[s]`, create an `exp_block` indexed by `k` in `{0, ..., n_exp-1}`.
    * **Structure:** `self.model.scen_block[s].exp_block[k]`.

3.  **Model Initialization Loop:**
    * **Outer Loop (Scenarios):** Iterate `s` from `0` to `Ns-1`.
    * **Inner Loop (Experiments):** Iterate `k` from `0` to `n_exp-1`.
        * **Step A (Model Creation):**
            * If the user provided `experiment_list`: Clone `experiment_list[k]`.
            * Else: Clone `self.experiment` and apply initialization values (e.g., via `set_exp_design_initial_values(k)`).
        * **Step B (Block Transfer):**
            * Transfer the attributes/variables from the cloned model to `self.model.scen_block[s].exp_block[k]`.
        * **Step C (Parameter Update - Future Hook):**
            * *Note:* If `Ns > 1`, update unknown parameters here using `parameter_scenarios[s]`. (Currently skipped).
        * **Step D (DoE Generation):**
            * Call `create_doe_model(self.model.scen_block[s].exp_block[k])` to generate Jacobians and FIM for this specific experiment block.
# Outer Loop: Reserved for uncertainty (currently runs once)
```python
for s in range(Ns):

    # Inner Loop: Simultaneous Experiments
    for k in range(n_exp):

        # Step A: Clone/Create the Model
        # If user provided experiment_list, clone experiment_list[k].
        # Else, clone self.experiment and apply initializers.
        current_model = ... 

        # Step B: Attach to Block Hierarchy
        # Note: We need to ensure the variables are properly transferred 
        # to the block, often using model.clone() or block assignment.
        self.model.scen_block[s].exp_block[k].transfer_attributes_from(current_model)

        # Step C: Generate Jacobians/FIM
        create_doe_model(self.model.scen_block[s].exp_block[k])
```
4.  **Symmetry Breaking (Simultaneous Design):**
    * **Condition:** If `n_exp > 1`:
    * Iterate `k` from `1` to `n_exp - 1`.
    * Retrieve the "primary" design variable (identified via `sym_break_cons` Suffix or user mapping).
    * Add Constraint:
        `self.model.scen_block[s].exp_block[k-1].primary_var <= self.model.scen_block[s].exp_block[k].primary_var`

5.  **Objective Function Construction:**
    * **Scenario FIM:** For each scenario `s`, compute the total FIM:
        $$
        \mathbf{M}_{total, s} = \sum_{k=0}^{n_{exp}-1} (\mathbf{M}_{k,s}) + \mathbf{M}_{prior}
        $$
    * **Objective Expression:** Define the maximizing objective:
        $$
        \max \sum_{s=0}^{N_s-1} w_s \cdot \log(\det(\mathbf{M}_{total, s}))
        $$

6.  **Solve:**
    * Send `self.model` to the solver (e.g., `ipopt`).