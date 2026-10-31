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
    # Data
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=["hour", "y"],
    )

    # Build experiment list
    exp_list = [RooneyBieglerExperiment(data.loc[i, :]) for i in range(data.shape[0])]

    # Create estimator
    pest = parmest.Estimator(exp_list, obj_function="SSE")

    # Compute profile likelihood for both unknown parameters.
    # Use a small grid for quick terminal runs.
    profiles, intervals, theta_hat, obj_hat = pest.profile_likelihood(
        profiled_theta=["asymptote", "rate_constant"],
        alphas=[0.9, 0.95],
        n_grid=9,
        warmstart="neighbor",
        # Demonstrate baseline from multistart integration:
        use_multistart_for_baseline=True,
        baseline_multistart_kwargs={
            "n_restarts": 5,
            "multistart_sampling_method": "uniform_random",
            "seed": 7,
        },
    )

    print("\nBaseline:")
    print(theta_hat, obj_hat)
    print("\nProfile of asymptote:")
    print(profiles.loc["asymptote", ["obj", "lr_stat", 0.95, "success"]])
    print("\nProfile confidence intervals:")
    print(intervals)

    # Plot the profiles with the threshold of each alpha
    parmest.graphics.profile_likelihood_plot(profiles, intervals)


if __name__ == "__main__":
    main()
