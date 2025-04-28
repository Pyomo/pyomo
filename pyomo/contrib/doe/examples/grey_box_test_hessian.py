#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.common.dependencies import numpy as np

from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import json
import logging
from pathlib import Path

from pyomo.contrib.doe import FIMExternalGreyBox


# Seeing if D-optimal experiment matches for both the
# greybox objective and the algebraic objective
def compare_reactor_doe():
    # Read in file
    DATA_DIR = Path(__file__).parent
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

    # solver = pyo.SolverFactory("ipopt")
    # solver.options["linear_solver"] = "mumps"

    # DoE object to compute FIM prior
    # doe_obj = DesignOfExperiments(
    #     experiment,
    #     fd_formula=fd_formula,
    #     step=step_size,
    #     objective_option=objective_option,
    #     use_grey_box_objective=True,  # New object with grey box set to True
    #     scale_constant_value=1,
    #     scale_nominal_param_value=scale_nominal_param_value,
    #     prior_FIM=None,
    #     jac_initial=None,
    #     fim_initial=None,
    #     L_diagonal_lower_bound=1e-7,
    #     solver=solver,
    #     tee=True,
    #     get_labeled_model_args=None,
    #     # logger_level=logging.ERROR,
    #     _Cholesky_option=True,
    #     _only_compute_fim_lower=True,
    # )

    # prior_FIM = doe_obj.compute_FIM(method="sequential")
    prior_FIM = None

    # Begin optimal grey box DoE
    ############################
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
        tee=True,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    doe_obj.run_doe()

    grey_box_check = FIMExternalGreyBox(doe_object=doe_obj)
    return grey_box_check, doe_obj


if __name__ == "__main__":
    compare_reactor_doe()
