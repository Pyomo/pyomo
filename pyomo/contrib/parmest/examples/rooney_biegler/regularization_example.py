# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.dependencies import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)


def main():
    """
    Evaluate L2 regularization on the Rooney-Biegler example.

    Notes
    -----
    The model response saturates for large positive ``rate_constant`` values:
    ``y = asymptote * (1 - exp(-rate_constant * hour))``.
    If ``rate_constant`` is both unpenalized and unbounded, the objective can be
    nearly flat in that direction, which can lead to extreme fitted values.
    """

    # Rooney & Biegler Reference Values
    # a = 19.14, b = 0.53
    theta_ref = {'asymptote': 19.14, 'rate_constant': 0.53}

    # L2 setup: create a 'Stiff' Prior for 'asymptote' but leave 'rate_constant' flexible.
    prior_FIM_l2 = pd.DataFrame(
        [[1000.0, 0.0], [0.0, 0.0]],
        index=['asymptote', 'rate_constant'],
        columns=['asymptote', 'rate_constant'],
    )

    # Data
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Create an experiment list
    exp_list = []
    for i in range(data.shape[0]):
        exp = RooneyBieglerExperiment(data.loc[i, :])
        # Example-scoped stabilization: keep rate_constant in a practical range.
        m = exp.get_labeled_model()
        m.rate_constant.setlb(0.0)
        m.rate_constant.setub(5.0)
        exp_list.append(exp)

    # Create an instance of the parmest estimator (L2)
    pest_l2 = parmest.Estimator(
        exp_list,
        obj_function="SSE",
        regularization='L2',
        prior_FIM=prior_FIM_l2,
        theta_ref=theta_ref,
    )

    # Parameter estimation and covariance for L2
    obj_l2, theta_l2 = pest_l2.theta_est()
    cov_l2 = pest_l2.cov_est()

    if parmest.graphics.seaborn_available:
        parmest.graphics.pairwise_plot(
            (theta_l2, cov_l2, 100),
            theta_star=theta_l2,
            alpha=0.8,
            distributions=['MVN'],
            title='L2 regularized theta estimates within 80% confidence region',
        )

    # Assert statements compare parameter estimation (theta) to an expected value
    relative_error = abs(theta_l2['asymptote'] - 19.1426) / 19.1426
    assert relative_error < 0.01
    relative_error = abs(theta_l2['rate_constant'] - 0.5311) / 0.5311
    assert relative_error < 0.01

    return {"L2": (obj_l2, theta_l2, cov_l2)}


if __name__ == "__main__":
    results = main()
    for reg_name, (obj, theta, cov) in results.items():
        print(f"{reg_name} estimated parameters (theta):", theta)
        print(f"{reg_name} objective function value at theta:", obj)
        print(f"{reg_name} covariance of parameter estimates:", cov)
