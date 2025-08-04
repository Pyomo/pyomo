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

from pyomo.common.dependencies import numpy as np, pandas as pd
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    ReactorDesignExperiment,
)

import pyomo.environ as pyo
from pyomo.contrib.parmest.utils.model_utils import update_model_from_suffix


def main():
    # Read in file
    # Read in data
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, "reactor_data.csv"))
    data = pd.read_csv(file_name)

    experiment = ReactorDesignExperiment(data, 0)

    # Call the experiment's model using get_labeled_model
    reactor_model = experiment.get_labeled_model()

    example_suffix = "unknown_parameters"
    suffix_obj = reactor_model.unknown_parameters
    var_list = list(suffix_obj.keys())  # components
    orig_var_vals = np.array([pyo.value(v) for v in var_list])  # numeric var values

    # Print original values
    print("Original sigma values")
    print("----------------------")
    print(orig_var_vals)

    # Update the suffix with new values
    new_vals = orig_var_vals + 0.5

    print("New sigma values")
    print("----------------")
    print(new_vals)

    # Here we are updating the values of the unknown parameters
    # You must know the length of the list and order of the suffix items to update them correctly
    update_model_from_suffix(suffix_obj, new_vals)

    # Updated values
    print("Updated sigma values :")
    print("-----------------------")
    new_var_vals = np.array([pyo.value(v) for v in var_list])
    print(new_var_vals)

    # Return the suffix obj, original and new values for further use if needed
    return suffix_obj, new_vals, new_var_vals


if __name__ == "__main__":
    main()
