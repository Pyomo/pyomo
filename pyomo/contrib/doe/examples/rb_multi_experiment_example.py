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
"""
Rooney-Biegler multi-experiment DoE example using LHS initialization.

This example uses a single template experiment and optimizes 2 experiments
simultaneously through ``optimize_experiments(n_exp=2, initialization_method="lhs")``.
"""

import pyomo.environ as pyo
from pathlib import Path

from pyomo.common.dependencies import pandas as pd
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.parmest.experiment import Experiment


def rooney_biegler_model(data, theta=None):
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {"asymptote": 15, "rate_constant": 0.5}

    model.asymptote = pyo.Var(initialize=theta["asymptote"])
    model.rate_constant = pyo.Var(initialize=theta["rate_constant"])
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
        self.model = rooney_biegler_model(
            self.data.to_frame().transpose(), theta=self.theta
        )

    def label_model(self):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data["y"])])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data["hour"])])

        # Use hour as symmetry-breaking variable for n_exp > 1.
        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def run_rb_multi_experiment_example(
    objective_option="determinant",
    lhs_n_samples=50,
    lhs_seed=7,
    tee=False,
    results_file=None,
):
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=["hour", "y"],
    )
    theta = {"asymptote": 15, "rate_constant": 0.5}
    measurement_error = 0.1

    template_exp = RooneyBieglerExperiment(
        data=data.loc[0, :], theta=theta, measure_error=measurement_error
    )

    doe_obj = DesignOfExperiments(
        experiment_list=[template_exp],
        objective_option=objective_option,
        tee=tee,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    if results_file is None:
        results_file = str(Path(__file__).with_name("rb_multi_experiment_results.json"))

    doe_obj.optimize_experiments(
        results_file=results_file,
        n_exp=2,
        initialization_method="lhs",
        lhs_n_samples=lhs_n_samples,
        lhs_seed=lhs_seed,
        lhs_parallel=True,
        lhs_combo_parallel=True,
        lhs_combo_chunk_size=500,
        lhs_combo_parallel_threshold=1000,
        lhs_max_wall_clock_time=1,
    )

    print("Solver Status:", doe_obj.results["Solver Status"])
    print("Termination Condition:", doe_obj.results["Termination Condition"])
    print("Initialization Method:", doe_obj.results["Initialization Method"])
    print("LHS Samples Per Dimension:", doe_obj.results["LHS Samples Per Dimension"])
    print("Results file:", results_file)

    for exp_idx, exp_result in enumerate(
        doe_obj.results["Scenarios"][0]["Experiments"]
    ):
        print(f"Experiment {exp_idx} design:", exp_result["Experiment Design"])

    return doe_obj


if __name__ == "__main__":
    doe_obj_solved = run_rb_multi_experiment_example(
        tee=False, objective_option="trace", lhs_n_samples=1000, lhs_seed=7
    )
    # print(doe_obj_solved.results)
