# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""
Rooney Biegler DoE example, based on Rooney, W. C. and Biegler, L. T. (2001). Design for
model parameter uncertainty using nonlinear confidence regions. AIChE Journal,
47(8), 1794-1804.
"""

from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)
from pyomo.contrib.doe import DesignOfExperiments, ObjectiveLib
from pyomo.common.dependencies import pandas as pd, numpy as np, matplotlib


def run_rooney_biegler_doe(
    sensitivity_formula="central",
    optimization_objective="determinant",
    improve_cholesky_roundoff_error=False,
    compute_FIM_full_factorial=False,
    draw_factorial_figure=False,
    design_range=None,
    tee=False,
    print_output=False,
):
    """Run Rooney Biegler DoE Example. All the boolean options control whether
    that section of the example is run. This is to facilitate testing using the example.

    Parameters
    ----------
    sensitivity_formula : str, optional
        Differentiation formula for computing the sensitivity matrix.
        Must be one of ['central', 'forward', 'backward']. By default 'central'.
        Note: symbolic differentiation is not currently supported for this example.
        Can be added in the future when that is implemented in Pyomo.DoE.
    optimization_objective : str or ObjectiveLib, optional
        Objective function for the design of experiments optimization.
        Can be a string ('determinant', 'trace', 'pseudo_trace', 'minimum_eigenvalue',
        'condition_number') or an ObjectiveLib enum member. By default 'determinant'.
    improve_cholesky_roundoff_error : bool, optional
        If True, applies additional constraints to improve Cholesky factorization
        round-off error. By default False.
    compute_FIM_full_factorial : bool, optional
        If True, computes the Fisher Information Matrix for all combinations in the
        full factorial design space defined by design_range. By default False.
    draw_factorial_figure : bool, optional
        If True, generates and saves factorial figures using the data from
        `compute_FIM_full_factorial` showing the design space exploration.
        By default False.
    design_range : dict
        Dictionary specifying the range of design variables to explore. Keys are
        variable names and values are lists of candidate values. Required for
        `compute_FIM_full_factorial`. By default {'hour': [0, 10, 40]}.
    tee : bool, optional
        If True, displays solver output during optimization. By default False.
    print_output : bool, optional
        If True, prints optimization results and full factorial design results
        to the console. By default False.

    Returns
    -------
    dict
        A dictionary containing:
        - 'experiment': The RooneyBieglerExperiment object used for design
        - 'results_dict': Full factorial design results (if computed)
        - 'optimization': Dictionary with optimization results including objective
          values and optimal designs
    """
    # Convert optimization_objective to enum if it's a string
    if isinstance(optimization_objective, str):
        optimization_objective = ObjectiveLib(optimization_objective)

    if design_range is None:
        design_range = {'hour': [0, 10, 40]}

    # Initialize a container for all potential results
    results_container = {
        "experiment": None,
        "results_dict": {},
        "optimization": {},  # Will hold D/A optimization results if run
    }

    # Data Setup
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )
    theta = {'asymptote': 15, 'rate_constant': 0.5}
    measurement_error = 0.1

    # Compute prior FIM from existing data
    FIM = np.zeros((2, 2))
    for i in range(len(data)):
        exp_i = RooneyBieglerExperiment(
            data=data.loc[i, :], theta=theta, measure_error=measurement_error
        )
        doe_prior = DesignOfExperiments(
            experiment=exp_i,
            fd_formula=sensitivity_formula,
            objective_option=optimization_objective,
            tee=tee,
        )
        FIM += doe_prior.compute_FIM()

    # Base Experiment for Design
    rooney_biegler_experiment = RooneyBieglerExperiment(
        data=data.loc[0, :], theta=theta, measure_error=measurement_error
    )
    results_container["experiment"] = rooney_biegler_experiment

    rooney_biegler_doe = DesignOfExperiments(
        experiment=rooney_biegler_experiment,
        objective_option=optimization_objective,
        fd_formula=sensitivity_formula,
        tee=tee,
        prior_FIM=FIM,
        improve_cholesky_roundoff_error=improve_cholesky_roundoff_error,
    )

    # Run the optimization
    rooney_biegler_doe.run_doe()

    # Store results based on objective type
    objective_name = optimization_objective.value
    results_container["optimization"] = {
        "objective_type": objective_name,
        "design": rooney_biegler_doe.results['Experiment Design'],
    }

    # Add objective-specific metrics
    if optimization_objective == ObjectiveLib.determinant:
        results_container["optimization"]["value"] = rooney_biegler_doe.results[
            'log10 D-opt'
        ]
    elif optimization_objective == ObjectiveLib.trace:
        results_container["optimization"]["value"] = rooney_biegler_doe.results[
            'log10 A-opt'
        ]
    elif optimization_objective == ObjectiveLib.pseudo_trace:
        results_container["optimization"]["value"] = rooney_biegler_doe.results[
            'log10 pseudo A-opt'
        ]
    elif optimization_objective == ObjectiveLib.minimum_eigenvalue:
        results_container["optimization"]["value"] = rooney_biegler_doe.results[
            'log10 E-opt'
        ]
    elif optimization_objective == ObjectiveLib.condition_number:
        results_container["optimization"]["value"] = rooney_biegler_doe.results[
            'FIM Condition Number'
        ]

    if print_output:
        print(f"Optimal results for {objective_name}:", rooney_biegler_doe.results)

    # Compute Full Factorial Design Results
    if compute_FIM_full_factorial:
        results_container["results_dict"] = (
            rooney_biegler_doe.compute_FIM_full_factorial(design_ranges=design_range)
        )
        if print_output:
            print("Full Factorial Design Results:\n", results_container["results_dict"])

    # Pyomo.DoE built-in factorial figure generation
    if draw_factorial_figure:
        rooney_biegler_doe.draw_factorial_figure(
            sensitivity_design_variables=['hour'],
            fixed_design_variables={},
            title_text="Rooney Biegler DoE - Full Factorial Design",
            xlabel_text="Experiment duration (hour)",
            log_scale=False,
            figure_file_name="rooney_biegler",
        )

    return results_container


if __name__ == "__main__":

    results = run_rooney_biegler_doe(
        optimization_objective=ObjectiveLib.trace,  # Can also use string: "trace"
        sensitivity_formula="central",
        improve_cholesky_roundoff_error=False,
        compute_FIM_full_factorial=True,
        draw_factorial_figure=True,  # Set True to test file generation
        design_range={'hour': [0, 10, 3]},  # Small range for speed
        print_output=True,
    )
    print(results)
    # Show plots if running locally
    matplotlib.pyplot.show()
