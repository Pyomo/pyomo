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

from pyomo.common.dependencies import pandas as pd
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

    # Parameter estimation with covariance
    obj, theta, cov = pest.theta_est(calc_cov=True, cov_n=17)
    print(obj)
    print(theta)


if __name__ == "__main__":
    main()
