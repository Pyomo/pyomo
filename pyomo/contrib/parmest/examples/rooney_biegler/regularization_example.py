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
    Compare L2 and smooth-L1 regularization on the Rooney-Biegler example.

    Notes
    -----
    The model response saturates for large positive ``rate_constant`` values:
    ``y = asymptote * (1 - exp(-rate_constant * hour))``.
    If ``rate_constant`` is both unpenalized and unbounded, the objective can be
    nearly flat in that direction, which can lead to extreme fitted values.
    To keep the smooth-L1 fit numerically stable and interpretable, this example:
    1) includes a nonzero L1 weight on ``rate_constant`` via ``prior_FIM_l1``, and
    2) applies finite bounds ``rate_constant in [0, 5]`` on each experiment model.
    """

    # Rooney & Biegler Reference Values
    # a = 19.14, b = 0.53
    theta_ref = pd.Series({'asymptote': 20.0, 'rate_constant': 0.8})

    # L2 setup: create a 'Stiff' Prior for 'asymptote' but leave 'rate_constant' flexible.
    prior_FIM_l2 = pd.DataFrame(
        [[1000.0, 0.0], [0.0, 0.0]],
        index=['asymptote', 'rate_constant'],
        columns=['asymptote', 'rate_constant'],
    )
    # L1 setup: penalize both parameters to avoid an unregularized flat direction.
    prior_FIM_l1 = pd.DataFrame(
        [[1000.0, 0.0], [0.0, 1.0]],
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

    # L1 smooth regularization uses sqrt((theta - theta_ref)^2 + epsilon)
    pest_l1 = parmest.Estimator(
        exp_list,
        obj_function="SSE",
        regularization='L1',
        prior_FIM=prior_FIM_l1,
        theta_ref=theta_ref,
        regularization_weight=1.0,
        regularization_epsilon=1e-6,
    )
    obj_l1, theta_l1 = pest_l1.theta_est()
    cov_l1 = pest_l1.cov_est()

    if parmest.graphics.seaborn_available:
        parmest.graphics.pairwise_plot(
            (theta_l2, cov_l2, 100),
            theta_star=theta_l2,
            alpha=0.8,
            distributions=['MVN'],
            title='L2 regularized theta estimates within 80% confidence region',
        )

    # Assert statements compare parameter estimation (theta) to an expected value
    # relative_error = abs(theta['asymptote'] - 19.1426) / 19.1426
    # assert relative_error < 0.01
    # relative_error = abs(theta['rate_constant'] - 0.5311) / 0.5311
    # assert relative_error < 0.01

    return {"L2": (obj_l2, theta_l2, cov_l2), "L1": (obj_l1, theta_l1, cov_l1)}


if __name__ == "__main__":
    results = main()
    for reg_name, (obj, theta, cov) in results.items():
        print(f"{reg_name} estimated parameters (theta):", theta)
        print(f"{reg_name} objective function value at theta:", obj)
        print(f"{reg_name} covariance of parameter estimates:", cov)
