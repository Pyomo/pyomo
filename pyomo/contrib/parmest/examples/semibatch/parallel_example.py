#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
The following script can be used to run semibatch parameter estimation in 
parallel and save results to files for later analysis and graphics.
Example command: mpiexec -n 4 python parallel_example.py
"""
import numpy as np
import pandas as pd
from itertools import product
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.semibatch.semibatch import generate_model


def main():
    # Vars to estimate
    theta_names = ['k1', 'k2', 'E1', 'E2']

    # Data, list of json file names
    data = []
    file_dirname = dirname(abspath(str(__file__)))
    for exp_num in range(10):
        file_name = abspath(join(file_dirname, 'exp' + str(exp_num + 1) + '.out'))
        data.append(file_name)

    # Note, the model already includes a 'SecondStageCost' expression
    # for sum of squared error that will be used in parameter estimation

    pest = parmest.Estimator(generate_model, data, theta_names)

    ### Parameter estimation with bootstrap resampling
    bootstrap_theta = pest.theta_est_bootstrap(100)
    bootstrap_theta.to_csv('bootstrap_theta.csv')

    ### Compute objective at theta for likelihood ratio test
    k1 = np.arange(4, 24, 3)
    k2 = np.arange(40, 160, 40)
    E1 = np.arange(29000, 32000, 500)
    E2 = np.arange(38000, 42000, 500)
    theta_vals = pd.DataFrame(list(product(k1, k2, E1, E2)), columns=theta_names)

    obj_at_theta = pest.objective_at_theta(theta_vals)
    obj_at_theta.to_csv('obj_at_theta.csv')


if __name__ == "__main__":
    main()
