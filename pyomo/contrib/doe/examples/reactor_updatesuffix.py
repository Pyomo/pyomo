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

    # Update the model to change the values of the desired component
    # Here we will update the unknown parameters of the reactor model
    example_suffix = "unknown_parameters"
    suffix_obj = reactor_model.unknown_parameters

    # Original values
    print(f"Original values of {example_suffix}: \n")
    for v in suffix_obj:
        v.display()  # prints “v : <value>”

    # Update the suffix with new values
    # Here we are updating the values of the unknown parameters
    # You must know the length of the list and order of the suffix items to update them correctly
    utils.update_model_from_suffix(suffix_obj, [1, 0.5, 0.1, 1])

    # Updated values
    print(f"\nUpdated values of {example_suffix}: \n")
    for v in suffix_obj:
        v.display()  # prints “v : <value>”

    # Show suffix is unchanged
    print(f"\nSuffix '{example_suffix}' is unchanged: \n")
    print({comp.name: tag for comp, tag in suffix_obj.items()})


if __name__ == "__main__":
    run_reactor_update_suffix_items()
