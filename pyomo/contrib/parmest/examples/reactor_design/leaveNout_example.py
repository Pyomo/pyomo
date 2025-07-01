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

from pyomo.common.dependencies import numpy as np, pandas as pd
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

    seed = 524  # Set seed for reproducibility
    np.random.seed(seed)  # Set seed for reproducibility

    # Create more data for the example
    N = 50
    df_std = data.std().to_frame().transpose()
    df_rand = pd.DataFrame(np.random.normal(size=N))
    df_sample = data.sample(N, replace=True).reset_index(drop=True)
    data = df_sample + df_rand.dot(df_std) / 10

    # Create an experiment list
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(ReactorDesignExperiment(data, i))

    # View one model
    # exp0_model = exp_list[0].get_labeled_model()
    # exp0_model.pprint()

    pest = parmest.Estimator(exp_list, obj_function='SSE')

    # Parameter estimation
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)

    ### Parameter estimation with 'leave-N-out'
    # Example use case: For each combination of data where one data point is left
    # out, estimate theta
    lNo_theta = pest.theta_est_leaveNout(1, seed=524)
    print(lNo_theta.head())

    parmest.graphics.pairwise_plot(lNo_theta, theta, seed=524)

    ### Leave one out/boostrap analysis
    # Example use case: leave 25 data points out, run 20 bootstrap samples with the
    # remaining points, determine if the theta estimate using the points left out
    # is inside or outside an alpha region based on the bootstrap samples, repeat
    # 5 times. Results are stored as a list of tuples, see API docs for information.
    lNo = 25
    lNo_samples = 5
    bootstrap_samples = 20
    dist = "MVN"
    alphas = [0.7, 0.8, 0.9]

    results = pest.leaveNout_bootstrap_test(
        lNo, lNo_samples, bootstrap_samples, dist, alphas, seed=524
    )

    # Plot results for a single value of alpha
    alpha = 0.8
    for i in range(lNo_samples):
        theta_est_N = results[i][1]
        bootstrap_results = results[i][2]
        parmest.graphics.pairwise_plot(
            bootstrap_results,
            theta_est_N,
            alpha,
            ["MVN"],
            title="Alpha: " + str(alpha) + ", " + str(theta_est_N.loc[0, alpha]),
            seed=524 + i,  # setting the seed for testing repeatability,
            # for typical use cases, this should not be set
        )

    # Extract the percent of points that are within the alpha region
    r = [results[i][1].loc[0, alpha] for i in range(lNo_samples)]
    percent_true = sum(r) / len(r)
    print(percent_true)


if __name__ == "__main__":
    main()
