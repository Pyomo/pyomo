# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""
Debug-oriented DoE example for optimize mode with advanced ``run_config``.

This example demonstrates how to inspect the assembled NLP initial point for a
DoE optimization objective (A-opt / trace) before the final optimization solve.
"""

import json

from pyomo.common.dependencies import pathlib

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment


def run_reactor_trace_debug_example(
    nfe=10, ncp=3, top_constraints=20, solve_final_model=False
):
    """
    Run trace-objective DoE with advanced inspection-oriented ``run_config``.

    Parameters
    ----------
    nfe : int, optional
        Number of finite elements for discretization in the reactor model.
    ncp : int, optional
        Number of collocation points for discretization in the reactor model.
    top_constraints : int, optional
        Number of most-violated constraints to report in inspection results.
    solve_final_model : bool, optional
        If False, assemble and initialize only (debug mode). If True, run the
        final optimization solve after inspection.

    Returns
    -------
    DesignOfExperiments
        Configured DoE object with results populated in ``doe_obj.results``.
    """
    # 1) Load the baseline reactor data used by the existing DoE examples.
    #    This keeps the debug workflow aligned with standard Pyomo.DoE demos.
    data_dir = pathlib.Path(__file__).parent
    file_path = data_dir / "result.json"
    with open(file_path) as f:
        data_ex = json.load(f)
    # JSON keys are strings; cast control-point times back to floats expected
    # by the ReactorExperiment constructor.
    data_ex["control_points"] = {
        float(k): v for k, v in data_ex["control_points"].items()
    }

    # 2) Build the experiment object that provides get_labeled_model().
    #    The DoE framework will generate finite-difference scenarios from this.
    experiment = ReactorExperiment(data=data_ex, nfe=nfe, ncp=ncp)

    # 3) Configure DoE for optimize mode with A-opt (trace) objective.
    #    Trace uses Cholesky-related variables/constraints and is a common case
    #    where users want initialization diagnostics.
    doe_obj = DesignOfExperiments(
        experiment=experiment,
        fd_formula="central",
        step=1e-3,
        objective_option="trace",
        scale_constant_value=1,
        scale_nominal_param_value=True,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=None,
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    # Advanced run config:
    # - scenario_solver_options:
    #     applied ONLY to scenario generation and square initialization solves.
    #     Keep this robust so prerequisite solves can converge.
    # - final_solver_options:
    #     applied ONLY to final optimize solve. Use this to do short probe runs
    #     (small max_iter) without affecting prerequisite scenario solves.
    # - final_solve:
    #     set False for "assemble-and-inspect" mode (no final optimization).
    # - inspection:
    #     enable structured residual reporting and choose number of constraints
    #     to print.
    run_config = {
        "scenario_solver_options": {"max_iter": 3000},
        "final_solver_options": {"max_iter": 200},
        "final_solve": solve_final_model,
        "inspection": {"enabled": True, "top_constraints": top_constraints},
    }

    # 4) Run DoE with advanced configuration.
    #    Results are stored in doe_obj.results.
    doe_obj.run_doe(run_config=run_config)

    # 5) Report top residuals at post-initialization. This is the key diagnostic
    #    view when debugging initial-point infeasibilities before the final solve.
    print("Solver Status:", doe_obj.results["Solver Status"])
    print("Termination Condition:", doe_obj.results["Termination Condition"])
    print("")
    print("Top residuals after initialization:")
    for row in doe_obj.results["Constraint Residuals"]["post_initialization"]:
        print(
            "  {name}: violation={viol:.3e} [{ctype}]".format(
                name=row["constraint_name"],
                viol=row["violation"],
                ctype=row["constraint_type"],
            )
        )

    if solve_final_model:
        # Optional post-final-stage report to compare initialization residuals
        # against residuals after the optimization phase starts/finishes.
        print("")
        print("Top residuals after final stage:")
        for row in doe_obj.results["Constraint Residuals"]["post_final_stage"]:
            print(
                "  {name}: violation={viol:.3e} [{ctype}]".format(
                    name=row["constraint_name"],
                    viol=row["violation"],
                    ctype=row["constraint_type"],
                )
            )

    return doe_obj


if __name__ == "__main__":
    # Default invocation is debug-first: assemble and inspect without final solve.
    # Set solve_final_model=True in the function call above to include final NLP.
    run_reactor_trace_debug_example()
