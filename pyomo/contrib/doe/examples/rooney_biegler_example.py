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
Rooney Biegler model, based on Rooney, W. C. and Biegler, L. T. (2001). Design for
model parameter uncertainty using nonlinear confidence regions. AIChE Journal,
47(8), 1794-1804.
"""
from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe.examples.rooney_biegler_experiment import (
    RooneyBieglerExperimentDoE,
)
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import json
import sys


# Example comparing Cholesky factorization
# (standard solve) with grey box objective
# solve for the log-deteriminant of the FIM
# (D-optimality)
def run_rooney_biegler_doe():
    # Create a RooneyBiegler Experiment
    experiment = RooneyBieglerExperimentDoE(data={'hour': 2, 'y': 10.3})

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-3

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    data = [[1, 8.3], [7, 19.8], [2, 10.3], [5, 15.6], [3, 19.0], [4, 16.0]]
    FIM_prior = np.zeros((2, 2))
    # Calculate prior using existing experiments
    for i in range(len(data)):
        if i > int(sys.argv[1]):
            break
        prev_experiment = RooneyBieglerExperimentDoE(
            data={'hour': data[i][0], 'y': data[i][1]}
        )
        doe_obj = DesignOfExperiments(
            prev_experiment,
            fd_formula=fd_formula,
            step=step_size,
            objective_option=objective_option,
            scale_nominal_param_value=scale_nominal_param_value,
            prior_FIM=None,
            tee=False,
        )

        FIM_prior += doe_obj.compute_FIM(method='sequential')

    if sys.argv[1] == 0:
        FIM_prior[0][0] += 1e-6
        FIM_prior[1][1] += 1e-6

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
        prior_FIM=FIM_prior,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=None,
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    # Begin optimal DoE
    ####################
    doe_obj.run_doe()

    # Print out a results summary
    print("Optimal experiment values: ")
    print(
        "\tOptimal measurement time: {:.2f}".format(
            doe_obj.results["Experiment Design"][0]
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

    # Begin optimal greybox DoE
    ############################
    doe_obj_gb = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        use_grey_box_objective=True,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=FIM_prior,
        tee=False,
    )

    doe_obj_gb.run_doe()

    print("Optimal experiment values: ")
    print(
        "\tOptimal measurement time: {:.2f}".format(
            doe_obj_gb.results["Experiment Design"][0]
        )
    )
    print("FIM at optimal design:\n {}".format(np.array(doe_obj_gb.results["FIM"])))
    print(
        "Objective value at optimal design: {:.2f}".format(
            np.log10(np.exp(pyo.value(doe_obj_gb.model.objective)))
        )
    )

    ############################
    # End optimal greybox DoE


if __name__ == "__main__":
    run_rooney_biegler_doe()
