#!/usr/bin/env python
"""Test multi-experiment optimization with prior FIM"""

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np


def rooney_biegler_model(data, theta=None):
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {'asymptote': 15, 'rate_constant': 0.5}

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate_constant'])

    model.asymptote.fix()
    model.rate_constant.fix()

    model.hour = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))
    model.hour.fix()

    model.y = pyo.Var(within=pyo.PositiveReals, initialize=data["y"].iloc[0])

    def response_rule(m):
        return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

    model.response_function = pyo.Constraint(rule=response_rule)

    return model


class RooneyBieglerExperiment(Experiment):
    def __init__(self, data, measure_error=None, theta=None):
        self.data = data
        self.model = None
        self.measure_error = measure_error
        self.theta = theta

    def create_model(self):
        data_df = self.data.to_frame().transpose()
        self.model = rooney_biegler_model(data_df, theta=self.theta)

    def label_model(self):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data['y'])])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data['hour'])])

        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


# Data Setup
data = pd.DataFrame(
    data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
    columns=['hour', 'y'],
)
theta = {'asymptote': 15, 'rate_constant': 0.5}
measurement_error = 0.1

print("Computing prior FIM from existing data...")
FIM_0 = np.zeros((2, 2))
for i in range(len(data)):
    exp_data = data.loc[i, :]
    exp = RooneyBieglerExperiment(
        data=exp_data, theta=theta, measure_error=measurement_error
    )
    doe_obj = DesignOfExperiments(
        experiment_list=exp, objective_option='determinant', prior_FIM=None, tee=False
    )
    FIM_0 += doe_obj.compute_FIM()

print("\nPrior FIM from existing data:")
print(FIM_0)
print(f"Det: {np.linalg.det(FIM_0)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(FIM_0)}")

# Test with run_doe (single experiment, should work)
print("\n" + "=" * 60)
print("Testing run_doe() with single experiment")
print("=" * 60)
exp = RooneyBieglerExperiment(
    data=data.loc[0, :], theta=theta, measure_error=measurement_error
)
doe_single = DesignOfExperiments(
    experiment_list=exp, objective_option='determinant', prior_FIM=FIM_0, tee=False
)
doe_single.run_doe()
print(f"Status: {doe_single.results['Solver Status']}")
print(f"FIM: \n{np.array(doe_single.results['FIM'])}")
print(f"Design: {doe_single.results['Experiment Design']}")

# Test with optimize_experiments (should give same result for 1 experiment)
print("\n" + "=" * 60)
print("Testing optimize_experiments() with single experiment")
print("=" * 60)
exp2 = RooneyBieglerExperiment(
    data=data.loc[0, :], theta=theta, measure_error=measurement_error
)
doe_multi = DesignOfExperiments(
    experiment_list=exp2, objective_option='determinant', prior_FIM=FIM_0, tee=False
)
doe_multi.optimize_experiments()
print(f"Status: {doe_multi.results['Solver Status']}")
total_fim = np.array(doe_multi.results['Scenarios'][0]['Total FIM'])
print(f"Total FIM: \n{total_fim}")
print(f"Total FIM is symmetric: {np.allclose(total_fim, total_fim.T)}")
print(f"Det: {np.linalg.det(total_fim)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(total_fim)}")

# Verify variable names are stored at top level (not repeated per scenario/experiment)
print(f"\nVariable names stored at top level (outermost dict):")
print(f"  Experiment Design Names: {doe_multi.results['Experiment Design Names']}")
print(f"  Experiment Output Names: {doe_multi.results['Experiment Output Names']}")
print(f"  Unknown Parameter Names: {doe_multi.results['Unknown Parameter Names']}")
print(f"  Measurement Error Names: {doe_multi.results['Measurement Error Names']}")

# Verify scenario-level data
scenario_0 = doe_multi.results['Scenarios'][0]
print(f"\nScenario 0 data (same for all experiments in scenario):")
print(f"  Unknown Parameters: {scenario_0['Unknown Parameters']}")

# Verify individual experiment data extraction using helper functions
exp_results = scenario_0['Experiments'][0]
print(f"\nExperiment 0 values (vary per experiment):")
print(f"  Experiment Design: {exp_results['Experiment Design']}")
print(f"  Experiment Outputs: {exp_results['Experiment Outputs']}")
print(f"  Measurement Error: {exp_results['Measurement Error']}")
print(f"  Has FIM: {'FIM' in exp_results}")
print(f"  Has Sensitivity Matrix: {'Sensitivity Matrix' in exp_results}")

# Verify proper data placement
print(f"\nVerifying proper data placement:")
print(f"  'Unknown Parameters' in scenario: {'Unknown Parameters' in scenario_0}")
print(f"  'Unknown Parameters' in exp: {'Unknown Parameters' in exp_results}")
print(
    f"  'Experiment Design Names' in scenario: {'Experiment Design Names' in scenario_0}"
)
print(f"  'Experiment Design Names' in exp: {'Experiment Design Names' in exp_results}")
print(f"  'Experiment Output Names' in exp: {'Experiment Output Names' in exp_results}")

# Compare with run_doe results
print(f"\n" + "=" * 60)
print("Comparison: optimize_experiments vs run_doe")
print("=" * 60)
print(f"FIM match: {np.allclose(total_fim, np.array(doe_single.results['FIM']))}")
print(
    f"Design match: {np.allclose(exp_results['Experiment Design'], doe_single.results['Experiment Design'])}"
)
