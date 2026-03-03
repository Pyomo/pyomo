# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.dependencies import numpy as np, pandas as pd
from itertools import product
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    ReactorDesignExperiment,
)


def main():

    # Read in data
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, "reactor_data.csv"))
    data = pd.read_csv(file_name)

    # Create an experiment list
    exp_list = [ReactorDesignExperiment(data, i) for i in range(data.shape[0])]

    # Solver options belong here (Ipopt options shown as example)
    solver_options = {
        "max_iter": 1000,
        "tol": 1e-6,
    }

    pest = parmest.Estimator(exp_list, obj_function="SSE", solver_options=solver_options)

    # Single-start estimation
    obj, theta = pest.theta_est()
    print("Single-start objective:", obj)
    print("Single-start theta:\n", theta)

    # Multistart estimation
    results_df, best_theta, best_obj = pest.theta_est_multistart(
        n_restarts=10,
        multistart_sampling_method="uniform_random",
        seed=42,
        save_results=False,  # True if you want CSV via file_name=
    )

    print("\nMultistart best objective:", best_obj)
    print("Multistart best theta:", best_theta)
    print("\nAll multistart results:")
    print(results_df)


if __name__ == "__main__":
    main()
