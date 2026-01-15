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
Parallel version of FIM computation for all pairwise experiment combinations.
Demonstrates embarrassingly parallel computation using multiprocessing.
"""

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np
import time
import json
from multiprocessing import Pool, cpu_count


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
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data['y'])])

        # Add unknown parameters as a suffix
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        # Add measurement error as a suffix
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        # Add hour as an experiment input
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data['hour'])])

        # For multiple experiments, we need to add symmetry breaking constraints
        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def compute_fim_pair(args):
    """
    Compute FIM for a single (i, j) pair.
    This function is designed to be called in parallel.

    Parameters
    ----------
    args : tuple
        (i, j, hour_i, hour_j, theta, measurement_error)

    Returns
    -------
    dict
        Results dictionary containing hour1, hour2, FIM_2, and log10_det
    """
    i, j, hour_i, hour_j, theta, measurement_error = args

    # Create data series for each hour value
    data_i = pd.Series({'hour': hour_i, 'y': 10.0})  # y value doesn't matter
    data_j = pd.Series({'hour': hour_j, 'y': 10.0})

    # Compute FIM for first experiment
    exp_obj_1 = RooneyBieglerExperiment(
        data=data_i, measure_error=measurement_error, theta=theta
    )
    doe_obj_1 = DesignOfExperiments(experiment_list=[exp_obj_1])
    FIM_1 = doe_obj_1.compute_FIM()

    # Compute FIM for second experiment with prior
    exp_obj_2 = RooneyBieglerExperiment(
        data=data_j, measure_error=measurement_error, theta=theta
    )
    doe_obj_2 = DesignOfExperiments(experiment_list=[exp_obj_2], prior_FIM=FIM_1)
    FIM_2 = doe_obj_2.compute_FIM()

    # Compute log10 determinant
    det_val = np.linalg.det(FIM_2)
    log10_det = float(np.log10(det_val)) if det_val > 0 else None

    return {
        'hour1': float(hour_i),
        'hour2': float(hour_j),
        'i': int(i),
        'j': int(j),
        'FIM_2': FIM_2.tolist(),  # Convert numpy array to list for JSON serialization
        'log10_det': log10_det,
    }


def run_serial(hours, theta, measurement_error):
    """Serial computation for comparison."""
    results = []
    start_time = time.time()

    for i, hour_i in enumerate(hours):
        for j, hour_j in enumerate(hours):
            result = compute_fim_pair((i, j, hour_i, hour_j, theta, measurement_error))
            results.append(result)

    elapsed = time.time() - start_time
    return results, elapsed


def run_parallel(hours, theta, measurement_error, n_cores=None):
    """Parallel computation using multiprocessing."""
    if n_cores is None:
        n_cores = cpu_count()

    # Create list of all (i, j) pairs to compute
    tasks = []
    for i, hour_i in enumerate(hours):
        for j, hour_j in enumerate(hours):
            tasks.append((i, j, hour_i, hour_j, theta, measurement_error))

    start_time = time.time()

    # Use multiprocessing Pool to compute in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_fim_pair, tasks)

    elapsed = time.time() - start_time
    return results, elapsed


if __name__ == "__main__":
    # Create hour data with dense sampling around optimal region for verification
    # Optimal from optimization: hour1≈1.90, hour2≈10.0
    hours_p1 = np.linspace(0.1, 1.7, 15)  # Before optimal: coarse
    hours_p2 = np.linspace(1.71, 2.1, 50)  # Around optimal hour1: dense
    hours_p3 = np.linspace(2.2, 9.4, 15)  # Between: coarse
    hours_p4 = np.linspace(9.41, 10, 25)  # Around optimal hour2: dense
    hours = np.concatenate((hours_p1, hours_p2, hours_p3, hours_p4))

    print(f"\nVerification Grid Setup:")
    print(f"  Total points: {len(hours)}")
    print(f"  Range: [{hours[0]:.2f}, {hours[-1]:.2f}]")
    print(f"  Dense around hour1≈1.90: [1.71, 2.10] with {len(hours_p2)} points")
    print(f"  Dense around hour2≈10.0: [9.41, 10.00] with {len(hours_p4)} points")
    print(f"  Total combinations: {len(hours)**2}")

    theta = {'asymptote': 15, 'rate_constant': 0.5}
    measurement_error = 0.1

    n_cores = cpu_count()

    print(f"\nComputing FIM for verification using {n_cores} cores...\n")

    # Run parallel computation only
    results_parallel, time_parallel = run_parallel(
        hours, theta, measurement_error, n_cores
    )

    print(f"Computation completed in {time_parallel:.2f} seconds\n")

    # Save results to JSON file
    output_file = "rooney_biegler_fim_verification.json"
    print(f"Saving results to {output_file}...")

    output_data = {
        'metadata': {
            'n_hours': len(hours),
            'hour_range': [float(hours[0]), float(hours[-1])],
            'total_computations': len(results_parallel),
            'theta': theta,
            'measurement_error': measurement_error,
            'n_cores': n_cores,
            'computation_time': time_parallel,
            -'grid_description': 'Dense sampling around optimal regions (1.71-2.10, 9.41-10), coarse elsewhere',
        },
        'results': results_parallel,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    file_size_mb = len(json.dumps(output_data)) / 1024 / 1024
    print(f"Results saved! File size: {file_size_mb:.2f} MB")

    # Quick analysis
    results_df = pd.DataFrame(results_parallel)
    valid_results = results_df[results_df['log10_det'].notna()].copy()

    print(f"\nValid combinations (det > 0): {len(valid_results)} / {len(results_df)}")

    if len(valid_results) > 0:
        best_idx = valid_results['log10_det'].idxmax()
        best = valid_results.loc[best_idx]
        print("\n" + "=" * 70)
        print("GRID SEARCH VERIFICATION OF OPTIMIZATION RESULT")
        print("=" * 70)
        print(
            f"Best design from grid: Hour1={best['hour1']:.4f}, Hour2={best['hour2']:.4f}"
        )
        print(f"                       log10(det) = {best['log10_det']:.4f}")
        print(f"\nOptimization result:   Hour1≈1.900, Hour2≈10.000")
        print(f"                       log10(det) ≈ 6.0280")
        print("=" * 70 + "\n")
