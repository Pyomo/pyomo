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


# Example to run a DoE on the reactor
def run_reactor_doe():
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
    # We will not be passing any prior information in this example.
    # We also will rely on the initialization routine within
    # the DesignOfExperiments class.
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

    # Make design ranges to compute the full factorial design
    design_ranges = {"CA[0]": [1, 5, 9], "T[0]": [300, 700, 9]}

    # Compute the full factorial design with the sequential FIM calculation
    doe_obj.compute_FIM_full_factorial(design_ranges=design_ranges, method="sequential")

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
        figure_file_name="example_reactor_compute_FIM",
        log_scale=False,
    )


if __name__ == "__main__":
    run_reactor_doe()
