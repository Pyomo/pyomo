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

import json
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.semibatch.semibatch import generate_model


def main():
    # Vars to estimate
    theta_names = ['k1', 'k2', 'E1', 'E2']

    # Data, list of dictionaries
    data = []
    file_dirname = dirname(abspath(str(__file__)))
    for exp_num in range(10):
        file_name = abspath(join(file_dirname, 'exp' + str(exp_num + 1) + '.out'))
        with open(file_name, 'r') as infile:
            d = json.load(infile)
            data.append(d)

    # Note, the model already includes a 'SecondStageCost' expression
    # for sum of squared error that will be used in parameter estimation

    pest = parmest.Estimator(generate_model, data, theta_names)

    obj, theta = pest.theta_est()
    print(obj)
    print(theta)


if __name__ == '__main__':
    main()
