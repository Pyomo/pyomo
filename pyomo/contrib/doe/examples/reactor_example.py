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
from pathlib import Path


# Example to run a DOE on the reactor
def run_reactor_doe():
    # Read in file
    DATA_DIR = Path(__file__).parent
    file_path = DATA_DIR / "result.json"

    f = open(file_path)
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
    objective_option = "det"
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
        L_initial=None,
        L_LB=1e-7,
        solver=None,
        tee=False,
        args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

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


if __name__ == "__main__":
    run_reactor_doe()
