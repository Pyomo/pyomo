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
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(ReactorDesignExperiment(data, i))

    # View one model
    # exp0_model = exp_list[0].get_labeled_model()
    # exp0_model.pprint()

    pest = parmest.Estimator(exp_list, obj_function='SSE')

    # Parameter estimation
    obj, theta = pest.theta_est()

    # Find the objective value at each theta estimate
    k1 = [0.8, 0.85, 0.9]
    k2 = [1.6, 1.65, 1.7]
    k3 = [0.00016, 0.000165, 0.00017]
    theta_vals = pd.DataFrame(list(product(k1, k2, k3)), columns=["k1", "k2", "k3"])
    obj_at_theta = pest.objective_at_theta(theta_vals)

    # Run the likelihood ratio test
    LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])

    # Plot results
    parmest.graphics.pairwise_plot(
        LR, theta, 0.9, title="LR results within 90% confidence region"
    )


if __name__ == "__main__":
    main()
