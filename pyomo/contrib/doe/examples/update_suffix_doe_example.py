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
from pyomo.common.dependencies import numpy as np

from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe import utils

from pyomo.contrib.parmest.utils.model_utils import update_model_from_suffix
from os.path import join, abspath, dirname

import pyomo.environ as pyo

import json


# Example to run a DoE on the reactor
def main():
    # Read in file
    file_dirname = dirname(abspath(str(__file__)))
    file_path = abspath(join(file_dirname, "result.json"))

    # Read in data
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
    # The suffix object 'measurement_error' stores measurement error values for each component.
    # Here, we retrieve the original values from the suffix for inspection.
    suffix_obj = reactor_model.measurement_error
    me_vars = list(suffix_obj.keys())  # components
    orig_vals = np.array(list(suffix_obj.values()))

    # Original values
    print("Original sigma values")
    print("-----------------------")
    suffix_obj.display()

    # Update the suffix with new values
    new_vals = orig_vals + 1
    # Here we are updating the values of the measurement error
    # You must know the length of the list and order of the suffix items to update them correctly
    update_model_from_suffix(suffix_obj, new_vals)

    # Updated values
    print("Updated sigma values :")
    print("-----------------------")
    suffix_obj.display()
    return suffix_obj, orig_vals, new_vals


if __name__ == "__main__":
    main()
