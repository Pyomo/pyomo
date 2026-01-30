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
Test script for multi-experiment optimization using the Rooney-Biegler example.
This script demonstrates the use of optimize_experiments() method.

Based on Rooney, W. C. and Biegler, L. T. (2001). Design for model parameter
uncertainty using nonlinear confidence regions. AIChE Journal, 47(8), 1794-1804.
"""

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np
import json
from pathlib import Path


def rooney_biegler_model(data, theta=None):
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {'asymptote': 15, 'rate_constant': 0.5}

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate_constant'])

    # Fix the unknown parameters
    model.asymptote.fix()
    model.rate_constant.fix()

    # Add the experiment inputs
    model.hour = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

    # Fix the experiment inputs
    model.hour.fix()

    # Add the response variable
    model.y = pyo.Var(within=pyo.PositiveReals, initialize=data["y"].iloc[0])

    def response_rule(m):
        return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

    model.response_function = pyo.Constraint(rule=response_rule)

    return model


class RooneyBieglerExperiment(Experiment):

    def __init__(self, data, measure_error=None, theta=None):
        self.data = data
        self.model = None
        self.measure_error = measure_error
        self.theta = theta

    def create_model(self):
        # rooney_biegler_model expects a dataframe
        data_df = self.data.to_frame().transpose()
        self.model = rooney_biegler_model(data_df, theta=self.theta)

    def label_model(self):

        m = self.model

        # Add experiment outputs as a suffix
        # Experiment outputs suffix is required for parmest
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data['y'])])

        # Add unknown parameters as a suffix
        # Unknown parameters suffix is required for both Pyomo.DoE and parmest
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        # Add measurement error as a suffix
        # Measurement error suffix is required for Pyomo.DoE and
        #  `cov` estimation in parmest
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        # Add hour as an experiment input
        # Experiment inputs suffix is required for Pyomo.DoE
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data['hour'])])

        # For multiple experiments, we need to add symmetry breaking constraints
        # to avoid identical models as a suffix `sym_break_cons`
        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def run_rooney_biegler_multi_experiment_doe(
    experiment_list, objective_option="determinant", prior_FIM=None, tee=False
):
    """
    Test multi-experiment optimization with the Rooney-Biegler example.

    Parameters
    ----------
    experiment_list : list
        List of RooneyBieglerExperiment objects to optimize simultaneously
    objective_option : str, optional
        Objective function option ('determinant', 'trace', or 'pseudo_trace'), by default 'determinant'
    prior_FIM : np.ndarray, optional
        Prior Fisher Information Matrix, by default None
    tee : bool, optional
        Whether to show solver output, by default False

    Returns
    -------
    DesignOfExperiments
        The DoE object containing optimization results
    """
    # Get number of experiments
    n_exp = len(experiment_list)

    # Create the DesignOfExperiments object
    print(f"Objective: {objective_option}")
    print(f"\n{'='*60}")
    print(f"Testing Multi-Experiment Optimization with {n_exp} experiments")
    print(f"{'='*60}\n")

    # Note: Not using prior_FIM to avoid numerical issues with this simple model
    # Also not passing a custom solver - let DoE use its default
    doe_obj = DesignOfExperiments(
        experiment_list=experiment_list,
        objective_option=objective_option,
        prior_FIM=prior_FIM,
        tee=tee,
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
        print(f"  log10 ME-opt: {scenario['FIM Condition Number']:.4f}")

        # Print each experiment design
        for exp_idx, exp in enumerate(scenario['Experiments']):
            print(f"\n  Experiment {exp_idx}:")
            print(f"    Design Variables:")
            for name, value in zip(
                doe_obj.results['Experiment Design Names'], exp['Experiment Design']
            ):
                print(f"      {name}: {value:.4f}")

    print(f"\n{'='*60}\n")

    return doe_obj


if __name__ == "__main__":
    # Data Setup
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )
    theta = {'asymptote': 15, 'rate_constant': 0.5}
    measurement_error = 0.1

    # Create solver for initialization
    solver = pyo.SolverFactory("ipopt")
    # Use default linear solver

    # Test with 2 experiments
    print("\n" + "=" * 60)
    print("Multi-Experiment Optimization Test")
    print("=" * 60)

    # Objective options to test
    objective_options = ['determinant', 'trace', 'pseudo_trace']

    # Get script directory for saving results
    script_dir = Path(__file__).parent

    # Dictionary to store all results
    all_results = {}

    # Use no prior FIM for this simple example
    # Using prior from existing data can cause numerical issues
    p_FIM = None

    for obj_option in objective_options:
        print(f"\n\n{'#'*70}")
        print(f"# Running with objective: {obj_option}")
        print(f"{'#'*70}")

        # Create multiple experiment objects with different hour values
        n_exp = 2
        experiment_list = []

        print(f"\nCreating and initializing {n_exp} experiments...")
        for i in range(n_exp):
            # Use different rows from the dataframe for each experiment
            exp_data = data.loc[i, :]
            exp = RooneyBieglerExperiment(
                data=exp_data, theta=theta, measure_error=measurement_error
            )
            experiment_list.append(exp)
            print(f"  Experiment {i+1} created (hour={exp_data['hour']:.1f})")

        # Run multi-experiment optimization
        doe_obj = run_rooney_biegler_multi_experiment_doe(
            experiment_list=experiment_list,
            objective_option=obj_option,
            prior_FIM=p_FIM,
            tee=False,
        )

        # Extract and save results
        results_summary = {
            'objective_option': obj_option,
            'solver_status': str(doe_obj.results['Solver Status']),
            'termination_condition': str(doe_obj.results['Termination Condition']),
            'n_scenarios': doe_obj.results['Number of Scenarios'],
            'n_experiments': doe_obj.results['Number of Experiments per Scenario'],
            'timing': {
                'build_time': doe_obj.results['Build Time'],
                'initialization_time': doe_obj.results['Initialization Time'],
                'solve_time': doe_obj.results['Solve Time'],
                'total_time': doe_obj.results['Wall-clock Time'],
            },
            'scenarios': [],
        }

        # Extract scenario results
        for s_idx, scenario in enumerate(doe_obj.results['Scenarios']):
            scenario_data = {
                'scenario_idx': s_idx,
                'log10_A_opt': scenario['log10 A-opt'],
                'log10_D_opt': scenario['log10 D-opt'],
                'log10_E_opt': scenario['log10 E-opt'],
                'FIM_condition_number': scenario['FIM Condition Number'],
                'experiments': [],
            }

            # Extract experiment designs
            for exp_idx, exp in enumerate(scenario['Experiments']):
                exp_data = {'experiment_idx': exp_idx, 'design_variables': {}}
                for name, value in zip(
                    doe_obj.results['Experiment Design Names'], exp['Experiment Design']
                ):
                    exp_data['design_variables'][name] = value
                scenario_data['experiments'].append(exp_data)

            results_summary['scenarios'].append(scenario_data)

        all_results[obj_option] = results_summary

        # Save individual results file
        output_file = script_dir / f'rooney_biegler_multiexp_{obj_option}.json'
        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Save combined results
    combined_file = script_dir / 'rooney_biegler_multiexp_all_objectives.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nCombined results saved to: {combined_file}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
