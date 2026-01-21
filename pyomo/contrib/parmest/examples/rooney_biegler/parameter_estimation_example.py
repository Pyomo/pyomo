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
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)


def main():

    # Data
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Create an experiment list
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

    # Create an instance of the parmest estimator
    pest = parmest.Estimator(exp_list, obj_function="SSE")

    # Parameter estimation and covariance
    obj, theta = pest.theta_est()
    cov = pest.cov_est()

    if parmest.graphics.seaborn_available:
        parmest.graphics.pairwise_plot(
            (theta, cov, 100),
            theta_star=theta,
            alpha=0.8,
            distributions=['MVN'],
            title='Theta estimates within 80% confidence region',
        )

    # Assert statements compare parameter estimation (theta) to an expected value
    relative_error = abs(theta['asymptote'] - 19.1426) / 19.1426
    assert relative_error < 0.01
    relative_error = abs(theta['rate_constant'] - 0.5311) / 0.5311
    assert relative_error < 0.01

    return obj, theta, cov


if __name__ == "__main__":
    obj, theta, cov = main()
    print("Estimated parameters (theta):", theta)
    print("Objective function value at theta:", obj)
    print("Covariance of parameter estimates:", cov)
