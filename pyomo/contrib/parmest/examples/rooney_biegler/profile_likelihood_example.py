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
    profile_results = pest.profile_likelihood(
        profiled_theta=["asymptote", "rate_constant"],
        n_grid=9,
        solver="ef_ipopt",
        warmstart="neighbor",
        # Demonstrate baseline from multistart integration:
        use_multistart_for_baseline=True,
        baseline_multistart_kwargs={
            "n_restarts": 5,
            "multistart_sampling_method": "uniform_random",
            "seed": 7,
        },
    )

    # Display a compact summary table
    profiles = profile_results["profiles"]
    print("\nBaseline:")
    print(profile_results["baseline"])
    print("\nProfile results (first 12 rows):")
    print(
        profiles[
            [
                "profiled_theta",
                "theta_value",
                "obj",
                "delta_obj",
                "lr_stat",
                "status",
                "success",
            ]
        ].head(12)
    )

    # Plot profile curves to file for terminal/non-GUI usage
    out_file = "rooney_biegler_profile_likelihood.png"
    parmest.graphics.profile_likelihood_plot(
        profile_results, alpha=0.95, filename=out_file
    )
    print(f"\nSaved plot to: {out_file}")


if __name__ == "__main__":
    main()
