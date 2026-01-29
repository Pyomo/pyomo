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
Test script for multi-experiment optimization using the Rooney-Biegler example.
This script demonstrates the use of optimize_experiments() method.

Based on Rooney, W. C. and Biegler, L. T. (2001). Design for model parameter
uncertainty using nonlinear confidence regions. AIChE Journal, 47(8), 1794-1804.
"""

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

    # Fix the unknown parameters
    model.asymptote.fix()
    model.rate_constant.fix()

    # Add the experiment inputs
    model.hour = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

    # Fix the experiment inputs
    model.hour.fix()

    # Add the response variable
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
        # rooney_biegler_model expects a dataframe
        data_df = self.data.to_frame().transpose()
        self.model = rooney_biegler_model(data_df, theta=self.theta)

    def label_model(self):

        m = self.model

        # Add experiment outputs as a suffix
        # Experiment outputs suffix is required for parmest
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data['y'])])

        # Add unknown parameters as a suffix
        # Unknown parameters suffix is required for both Pyomo.DoE and parmest
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        # Add measurement error as a suffix
        # Measurement error suffix is required for Pyomo.DoE and
        #  `cov` estimation in parmest
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        # Add hour as an experiment input
        # Experiment inputs suffix is required for Pyomo.DoE
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data['hour'])])

        # For multiple experiments, we need to add symmetry breaking constraints
        # to avoid identical models as a suffix `sym_break_cons`
        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


if __name__ == "__main__":
    # Data Setup
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )
    theta = {'asymptote': 15, 'rate_constant': 0.5}
    measurement_error = 0.1

    # Storage for results
    results = []

    print("\n" + "=" * 80)
    print("Computing FIM for all pairwise experiment combinations")
    print("=" * 80)
    print(f"Total computations: {len(data)} x {len(data)} = {len(data)**2}")
    print(
        f"This is an EMBARRASSINGLY PARALLEL problem - each (i,j) pair is independent!\n"
    )

    for i in range(len(data)):
        hour1 = data.iloc[i]['hour']
        exp_obj_1 = RooneyBieglerExperiment(
            data=data.iloc[i], measure_error=measurement_error, theta=theta
        )
        doe_obj_1 = DesignOfExperiments(experiment_list=[exp_obj_1])
        FIM_1 = doe_obj_1.compute_FIM()

        for j in range(len(data)):
            hour2 = data.iloc[j]['hour']
            exp_obj_2 = RooneyBieglerExperiment(
                data=data.iloc[j], measure_error=measurement_error, theta=theta
            )
            doe_obj_2 = DesignOfExperiments(
                experiment_list=[exp_obj_2], prior_FIM=FIM_1
            )
            FIM_2 = doe_obj_2.compute_FIM()

            log10_det2 = np.log10(np.linalg.det(FIM_2))

            # Store results
            results.append(
                {
                    'hour1': hour1,
                    'hour2': hour2,
                    'i': i,
                    'j': j,
                    'FIM_2': FIM_2.copy(),
                    'log10_det': log10_det2,
                }
            )

            print(
                f"  [{i},{j}] hour1={hour1:.1f}, hour2={hour2:.1f}, log10(det(FIM))={log10_det2:.4f}"
            )

    # Convert results to a structured format for analysis
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)

    # Create a results DataFrame
    results_df = pd.DataFrame(results)
    print(f"\nTotal combinations computed: {len(results_df)}")
    print(f"\nBest design (highest log10 determinant):")
    best_idx = results_df['log10_det'].idxmax()
    best = results_df.loc[best_idx]
    print(f"  Hour 1: {best['hour1']:.1f}, Hour 2: {best['hour2']:.1f}")
    print(f"  log10(det(FIM)): {best['log10_det']:.4f}")

    print(f"\n{'='*80}")
    print("PARALLELIZATION ANALYSIS")
    print(f"{'='*80}")
    print("\nThis problem is EMBARRASSINGLY PARALLEL because:")
    print("  1. Each (i,j) pair computation is completely independent")
    print("  2. No communication needed between computations")
    print("  3. Results can be collected after all computations complete")
    print("\nParallelization strategies:")
    print("  - Option 1: Python multiprocessing.Pool with map()")
    print("  - Option 2: joblib.Parallel with n_jobs=-1")
    print("  - Option 3: Dask for distributed computing")
    print("  - Option 4: Ray for more complex workflows")
    print(f"\nExpected speedup with {len(data)**2} tasks on N cores: ~N-fold (ideal)")
    print(f"{'='*80}\n")
