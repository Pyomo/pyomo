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
from pyomo.contrib.doe import utils

from pyomo.contrib.parmest.utils.model_utils import update_model_from_suffix

import pyomo.environ as pyo

import json


# Example to run a DoE on the reactor
def run_reactor_update_suffix_items():
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

    # Call the experiment's model using get_labeled_model
    reactor_model = experiment.get_labeled_model()

    # Show the model
    reactor_model.pprint()
    # Update the model to change the values of the desired component
    # Here we will update the unknown parameters of the reactor model
    example_suffix = "measurement_error"
    suffix_obj = reactor_model.measurement_error
    me_vars = list(suffix_obj.keys())  # components
    orig_vals = np.array([suffix_obj[v] for v in me_vars])

    # Original values
    print("Original sigma values:", orig_vals)
    # Update the suffix with new values
    new_vals = orig_vals + 1
    # Here we are updating the values of the unknown parameters
    # You must know the length of the list and order of the suffix items to update them correctly
    update_model_from_suffix(suffix_obj, new_vals)

    # Updated values
    print("Updated sigma values :", [suffix_obj[v] for v in me_vars])


if __name__ == "__main__":
    run_reactor_update_suffix_items()
