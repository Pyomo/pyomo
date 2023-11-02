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
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    reactor_design_model,
)


def main():
    # Vars to estimate
    theta_names = ["k1", "k2", "k3"]

    # Data
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, "reactor_data.csv"))
    data = pd.read_csv(file_name)

    # Sum of squared error function
    def SSE(model, data):
        expr = (
            (float(data.iloc[0]["ca"]) - model.ca) ** 2
            + (float(data.iloc[0]["cb"]) - model.cb) ** 2
            + (float(data.iloc[0]["cc"]) - model.cc) ** 2
            + (float(data.iloc[0]["cd"]) - model.cd) ** 2
        )
        return expr

    # Create an instance of the parmest estimator
    pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)

    # Parameter estimation
    obj, theta = pest.theta_est()

    # Parameter estimation with bootstrap resampling
    bootstrap_theta = pest.theta_est_bootstrap(50)

    # Plot results
    parmest.graphics.pairwise_plot(bootstrap_theta, title="Bootstrap theta")
    parmest.graphics.pairwise_plot(
        bootstrap_theta,
        theta,
        0.8,
        ["MVN", "KDE", "Rect"],
        title="Bootstrap theta with confidence regions",
    )


if __name__ == "__main__":
    main()
