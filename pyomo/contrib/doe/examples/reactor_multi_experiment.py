#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
Test script for multi-experiment optimization using the reactor example.
This script demonstrates the use of optimize_experiments() method.
"""
from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo
import json


def run_reactor_multi_experiment_doe(experiment_list, tee=False):
    """
    Test multi-experiment optimization with the reactor example.

    Parameters
    ----------
    experiment_list : list
        List of ReactorExperiment objects to optimize simultaneously
    tee : bool, optional
        Whether to show solver output, by default False
    """
    # Read in file
    DATA_DIR = pathlib.Path(__file__).parent
    file_path = DATA_DIR / "result.json"

    with open(file_path) as f:
        data_ex = json.load(f)

    # Put temperature control time points into correct format for reactor experiment
    data_ex["control_points"] = {
        float(k): v for k, v in data_ex["control_points"].items()
    }

    # Get number of experiments
    n_exp = len(experiment_list)

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-3

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    # Create the DesignOfExperiments object
    print(f"\n{'='*60}")
    print(f"Testing Multi-Experiment Optimization with {n_exp} experiments")
    print(f"{'='*60}\n")

    # Create a custom solver
    solver = pyo.SolverFactory("ipopt")
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["max_iter"] = 3000

    doe_obj = DesignOfExperiments(
        experiment_list=experiment_list,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=solver,  # Use custom solver
        tee=tee,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    # Run multi-experiment optimization
    print(f"Running optimize_experiments with {n_exp} experiments...")
    doe_obj.optimize_experiments()

    # Print results
    print(f"\n{'='*60}")
    print("Multi-Experiment Optimization Results")
    print(f"{'='*60}\n")

    print(f"Solver Status: {doe_obj.results['Solver Status']}")
    print(f"Termination Condition: {doe_obj.results['Termination Condition']}")
    print(f"Number of Scenarios: {doe_obj.results['Number of Scenarios']}")
    print(
        f"Number of Experiments per Scenario: {doe_obj.results['Number of Experiments per Scenario']}"
    )

    print(f"\nTiming:")
    print(f"  Build Time: {doe_obj.results['Build Time']:.2f} seconds")
    print(
        f"  Initialization Time: {doe_obj.results['Initialization Time']:.2f} seconds"
    )
    print(f"  Solve Time: {doe_obj.results['Solve Time']:.2f} seconds")
    print(f"  Total Time: {doe_obj.results['Wall-clock Time']:.2f} seconds")

    # Print scenario-specific results
    for s_idx, scenario in enumerate(doe_obj.results['Scenarios']):
        print(f"\n{'-'*60}")
        print(f"Scenario {s_idx} Results:")
        print(f"{'-'*60}")

        print(f"\nAggregated FIM Statistics:")
        print(f"  log10 A-opt: {scenario['log10 A-opt']:.4f}")
        print(f"  log10 D-opt: {scenario['log10 D-opt']:.4f}")
        print(f"  log10 E-opt: {scenario['log10 E-opt']:.4f}")
        print(f"  FIM Condition Number: {scenario['FIM Condition Number']:.4f}")

        # Print each experiment design
        for exp_idx, exp in enumerate(scenario['Experiments']):
            print(f"\n  Experiment {exp_idx}:")
            print(f"    Design Variables:")
            for name, value in zip(
                exp['Experiment Design Names'], exp['Experiment Design']
            ):
                print(f"      {name}: {value:.4f}")

    print(f"\n{'='*60}\n")

    return doe_obj


if __name__ == "__main__":
    # Load experiment data
    DATA_DIR = pathlib.Path(__file__).parent
    file_path = DATA_DIR / "result.json"
    with open(file_path) as f:
        data_ex = json.load(f)
    data_ex["control_points"] = {
        float(k): v for k, v in data_ex["control_points"].items()
    }

    # Test with 1 experiment first (simplest case)
    print("\n" + "=" * 60)
    print("TEST 1: Single Experiment")
    print("=" * 60)
    # Create a FRESH experiment object each time
    exp1 = ReactorExperiment(data=data_ex.copy(), nfe=10, ncp=3)
    solver = pyo.SolverFactory("ipopt")
    solver.solve(exp1.get_labeled_model())  # Initial solve to help with initialization
    doe_obj_1 = run_reactor_multi_experiment_doe(experiment_list=[exp1], tee=False)

    # Test with 2 experiments with different initial values
    print("\n" + "=" * 60)
    print("TEST 2: Two Experiments")
    print("=" * 60)

    # Create data for first experiment - use base values
    data_ex1 = data_ex.copy()
    data_ex1["control_points"] = {
        float(k): v for k, v in data_ex["control_points"].items()
    }
    # CA0 will be different for each experiment
    data_ex1["CA0"] = 2.0  # Different initial concentration

    # Create data for second experiment - perturb temperature control points
    data_ex2 = data_ex.copy()
    data_ex2["control_points"] = {
        float(k): v for k, v in data_ex["control_points"].items()
    }
    data_ex2["CA0"] = 2.5  # Different initial concentration

    exp2_1 = ReactorExperiment(data=data_ex1, nfe=10, ncp=3)
    exp2_2 = ReactorExperiment(data=data_ex2, nfe=10, ncp=3)

    doe_obj_2 = run_reactor_multi_experiment_doe(
        experiment_list=[exp2_1, exp2_2], tee=False
    )

    print("\nAll tests completed successfully!")
