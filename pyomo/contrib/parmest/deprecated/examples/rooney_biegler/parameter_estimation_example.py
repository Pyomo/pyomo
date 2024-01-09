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

import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    rooney_biegler_model,
)


def main():
    # Vars to estimate
    theta_names = ['asymptote', 'rate_constant']

    # Data
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Sum of squared error function
    def SSE(model, data):
        expr = sum(
            (data.y[i] - model.response_function[data.hour[i]]) ** 2 for i in data.index
        )
        return expr

    # Create an instance of the parmest estimator
    pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE)

    # Parameter estimation and covariance
    n = 6  # total number of data points used in the objective (y in 6 scenarios)
    obj, theta, cov = pest.theta_est(calc_cov=True, cov_n=n)

    # Plot theta estimates using a multivariate Gaussian distribution
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


if __name__ == "__main__":
    main()
