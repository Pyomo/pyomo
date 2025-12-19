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

from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np
import matplotlib.pyplot as plt

theta = {'asymptote': 15, 'rate_constant': 0.5}
measurement_error = 0.1

# Data
data = pd.DataFrame(
    data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
    columns=['hour', 'y'],
)
time_points = np.linspace(0, 10, 40)

results_dict = {"time_points": [], "log10 D_opt": [], "log10 A_opt": []}

# Create the model
first_data_point = data.loc[0, :]
rooney_biegler_experiment = RooneyBieglerExperiment(
    data=first_data_point, theta=theta, measure_error=measurement_error
)
rooney_biegler_experiment.get_labeled_model()
model = rooney_biegler_experiment.model

FIM = np.zeros((2, 2))
for i in range(len(data)):
    rooney_biegler_experiment = RooneyBieglerExperiment(
        data=data.loc[i, :], theta=theta, measure_error=measurement_error
    )
    rooney_biegler_doe = DesignOfExperiments(
        experiment=rooney_biegler_experiment, objective_option="determinant", tee=False
    )
    FIM_i = rooney_biegler_doe.compute_FIM()
    FIM += FIM_i
# Optimize the experiments
rooney_biegler_experiment = RooneyBieglerExperiment(
    data=data.loc[0, :], theta=theta, measure_error=measurement_error
)
rooney_biegler_doe_D = DesignOfExperiments(
    experiment=rooney_biegler_experiment,
    objective_option="determinant",
    tee=False,
    prior_FIM=FIM,
)
rooney_biegler_doe_D.run_doe()
opt_val_D = rooney_biegler_doe_D.results['log10 D-opt']
opt_design_D = rooney_biegler_doe_D.results['Experiment Design']


rooney_biegler_doe_A = DesignOfExperiments(
    experiment=rooney_biegler_experiment,
    objective_option="trace",
    tee=False,
    prior_FIM=FIM,
)
rooney_biegler_doe_A.run_doe()
opt_val_A = rooney_biegler_doe_A.results['log10 A-opt']
opt_design_A = rooney_biegler_doe_A.results['Experiment Design']

rooney_biegler_doe = DesignOfExperiments(
    experiment=rooney_biegler_experiment,
    objective_option="determinant",
    tee=False,
    prior_FIM=FIM,
)
for time_point in time_points:
    # The output variable is not need to be passed for DesignOfExperiments
    rooney_biegler_doe.experiment.model.hour.set_value(time_point)

    rooney_biegler_doe.experiment = rooney_biegler_experiment

    factorial_FIM = rooney_biegler_doe.compute_FIM()
    results_dict["time_points"].append(time_point)
    results_dict["log10 D_opt"].append(np.log10(np.linalg.det(factorial_FIM)))
    results_dict["log10 A_opt"].append(np.log10(np.trace(np.linalg.inv(factorial_FIM))))

# Plotting the results
id_max_D = np.argmax(results_dict["log10 D_opt"])
star_value_D = results_dict["log10 D_opt"][id_max_D]
star_values_time_D = results_dict["time_points"][id_max_D]

id_min_A = np.argmin(results_dict["log10 A_opt"])
star_values_time_A = results_dict["time_points"][id_min_A]
star_value_A = results_dict["log10 A_opt"][id_min_A]
plt.figure(figsize=(10, 5))
plt.plot(results_dict["time_points"], results_dict["log10 D_opt"])
plt.scatter(
    opt_design_D[0],
    opt_val_D,
    color='green',
    marker="*",
    s=600,
    label=f'Optimal D-optimality: {opt_val_D:.2f} at t={opt_design_D[0]:.2f}',
)
plt.scatter(
    star_values_time_D,
    star_value_D,
    color='red',
    marker="o",
    s=200,
    label=f'D-optimality: {star_value_D:.2f} at t={star_values_time_D:.2f}',
)

plt.legend()
plt.xlabel("Time Points")
plt.ylabel("log10 D-optimality")
plt.title("D-optimality vs Time Points")

plt.figure(figsize=(10, 5))
plt.plot(results_dict["time_points"], results_dict["log10 A_opt"])
plt.scatter(
    opt_design_A[0],
    opt_val_A,
    color='green',
    marker="*",
    s=600,
    label=f'Optimal A-optimality: {opt_val_A:.2f} at t={opt_design_A[0]:.2f}',
)
plt.scatter(
    star_values_time_A,
    star_value_A,
    color='red',
    marker="o",
    s=200,
    label=f'A-optimality: {star_value_A:.2f} at t={star_values_time_A:.2f}',
)

plt.legend()
plt.xlabel("Time Points")
plt.ylabel("log10 A-optimality")
plt.show()
