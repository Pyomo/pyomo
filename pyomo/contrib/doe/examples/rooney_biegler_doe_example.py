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
Rooney Biegler DoE example, based on Rooney, W. C. and Biegler, L. T. (2001). Design for
model parameter uncertainty using nonlinear confidence regions. AIChE Journal,
47(8), 1794-1804.
"""

from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np, matplotlib


def run_rooney_biegler_doe(
    optimize_experiment_A=False,
    optimize_experiment_D=False,
    compute_FIM_full_factorial=False,
    draw_factorial_figure=False,
    design_range={'hour': [0, 10, 40]},
    tee=False,
    print_output=False,
):
    """Run Rooney Biegler DoE Example. All the boolean options control whether
    that section of the example is run. This is to facilitate testing using the example.

    Parameters
    ----------
    optimize_experiment_A : bool, optional
        If True, performs A-optimality optimization to find the optimal experimental
        design that minimizes the trace of the inverse Fisher Information Matrix.
        By default False.
    optimize_experiment_D : bool, optional
        If True, performs D-optimality optimization to find the optimal
        experimental design that maximizes the determinant of the Fisher Information
        Matrix. By default False.
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
        - 'optimization': Dictionary with 'D' and/or 'A' keys containing optimization
          results including objective values and optimal designs
        - 'plots': List of matplotlib figure objects (if plots were generated)
    """
    # Initialize a container for all potential results
    results_container = {
        "experiment": None,
        "results_dict": {},
        "optimization": {},  # Will hold D/A optimization results if run
        "plots": [],  # Store figure objects if created
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
            experiment=exp_i, objective_option="determinant", tee=tee
        )
        FIM += doe_prior.compute_FIM()

    # Base Experiment for Design
    rooney_biegler_experiment = RooneyBieglerExperiment(
        data=data.loc[0, :], theta=theta, measure_error=measurement_error
    )
    results_container["experiment"] = rooney_biegler_experiment

    rooney_biegler_doe = DesignOfExperiments(
        experiment=rooney_biegler_experiment,
        objective_option="determinant",
        tee=tee,
        prior_FIM=FIM,
        improve_cholesky_roundoff_error=True,
    )

    if optimize_experiment_D:
        rooney_biegler_doe.run_doe()

        results_container["optimization"]["D"] = {
            "value": rooney_biegler_doe.results['log10 D-opt'],
            "design": rooney_biegler_doe.results['Experiment Design'],
        }
        if print_output:
            print("Optimal results for D-optimality:", rooney_biegler_doe.results)

    if optimize_experiment_A:
        # A-Optimality
        rooney_biegler_doe_A = DesignOfExperiments(
            experiment=rooney_biegler_experiment,
            objective_option="trace",
            tee=tee,
            prior_FIM=FIM,
            improve_cholesky_roundoff_error=False,
        )
        rooney_biegler_doe_A.run_doe()

        results_container["optimization"]["A"] = {
            "value": rooney_biegler_doe_A.results['log10 A-opt'],
            "design": rooney_biegler_doe_A.results['Experiment Design'],
        }

        if print_output:
            print("Optimal results for A-optimality:", rooney_biegler_doe_A.results)

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
            log_scale=False,
            figure_file_name="rooney_biegler",
        )

    return results_container


if __name__ == "__main__":

    results = run_rooney_biegler_doe(
        optimize_experiment_A=True,
        optimize_experiment_D=True,
        compute_FIM_full_factorial=True,
        draw_factorial_figure=True,  # Set True to test file generation
        design_range={'hour': [0, 10, 3]},  # Small range for speed
        print_output=True,
    )
    print(results)
    # Show plots if running locally
    # matplotlib.pyplot.show()
