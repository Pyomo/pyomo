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
from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import json


# Example for sensitivity analysis on the reactor experiment
# After sensitivity analysis is done, we perform optimal DoE
def run_reactor_doe(
    n_points_for_design=9,
    compute_FIM_full_factorial=True,
    plot_factorial_results=True,
    save_plots=True,
    run_optimal_doe=True,
):
    """
    This function demonstrates how to perform sensitivity analysis on the reactor

    Parameters
    ----------
    n_points_for_design : int, optional
        number of points to use for the design ranges, by default 9
    compute_FIM_full_factorial : bool, optional
        whether to compute the full factorial design, by default True
    plot_factorial_results : bool, optional
        whether to plot the results of the full factorial design, by default True
    save_plots : bool, optional
        whether to save draw_factorial_figure plots, by default True
    run_optimal_doe : bool, optional
        whether to run the optimal DoE, by default True
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

    # Create a ReactorExperiment object; data and discretization information are part
    # of the constructor of this object
    experiment = ReactorExperiment(data=data_ex, nfe=10, ncp=3)

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-3

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    # Create the DesignOfExperiments object
    # We will not be passing any prior information in this example
    # and allow the experiment object and the DesignOfExperiments
    # call of ``run_doe`` perform model initialization.
    doe_obj = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
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
    if compute_FIM_full_factorial:
        # Make design ranges to compute the full factorial design
        design_ranges = {
            "CA[0]": [1, 5, n_points_for_design],
            "T[0]": [300, 700, n_points_for_design],
        }

        # Compute the full factorial design with the sequential FIM calculation
        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )
    if plot_factorial_results:
        if save_plots:
            figure_file_name = "example_reactor_compute_FIM"
        else:
            figure_file_name = None

        # Plot the results
        doe_obj.draw_factorial_figure(
            sensitivity_design_variables=["CA[0]", "T[0]"],
            fixed_design_variables={
                "T[0.125]": 300,
                "T[0.25]": 300,
                "T[0.375]": 300,
                "T[0.5]": 300,
                "T[0.625]": 300,
                "T[0.75]": 300,
                "T[0.875]": 300,
                "T[1]": 300,
            },
            title_text="Reactor Example",
            xlabel_text="Concentration of A (M)",
            ylabel_text="Initial Temperature (K)",
            figure_file_name=figure_file_name,
            log_scale=False,
        )

    ###########################
    # End sensitivity analysis

    # Begin optimal DoE
    ####################
    if run_optimal_doe:
        doe_obj.run_doe()

        # Print out a results summary
        print("Optimal experiment values: ")
        print(
            "\tInitial concentration: {:.2f}".format(
                doe_obj.results["Experiment Design"][0]
            )
        )
        print(
            ("\tTemperature values: [" + "{:.2f}, " * 8 + "{:.2f}]").format(
                *doe_obj.results["Experiment Design"][1:]
            )
        )
        print("FIM at optimal design:\n {}".format(np.array(doe_obj.results["FIM"])))
        print(
            "Objective value at optimal design: {:.2f}".format(
                pyo.value(doe_obj.model.objective)
            )
        )

        print(doe_obj.results["Experiment Design Names"])

        ###################
        # End optimal DoE

    return doe_obj


if __name__ == "__main__":
    run_reactor_doe()
