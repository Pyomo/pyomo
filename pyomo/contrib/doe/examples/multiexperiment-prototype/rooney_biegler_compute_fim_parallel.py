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
from pathlib import Path


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


def compute_fim_combination(args):
    """
    Compute FIM for a combination of experiments.
    This function is designed to be called in parallel.

    Parameters
    ----------
    args : tuple
        (indices, hours, theta, measurement_error)
        where indices is a tuple of experiment indices,
        and hours is a tuple of corresponding hour values

    Returns
    -------
    dict
        Results dictionary containing hour values, indices, FIM, and log10_det
    """
    indices, hours, theta, measurement_error = args

    # Compute FIM sequentially, using each previous FIM as prior
    prior_FIM = None
    for idx, hour_val in enumerate(hours):
        # Create data series for this hour value
        data = pd.Series({'hour': hour_val, 'y': 10.0})  # y value doesn't matter

        # Create experiment and DOE object
        exp_obj = RooneyBieglerExperiment(
            data=data, measure_error=measurement_error, theta=theta
        )
        doe_obj = DesignOfExperiments(experiment_list=[exp_obj], prior_FIM=prior_FIM)
        prior_FIM = doe_obj.compute_FIM()

    final_FIM = prior_FIM

    # Compute log10 determinant
    det_val = np.linalg.det(final_FIM)
    log10_det = float(np.log10(det_val)) if det_val > 0 else None

    # Build result dictionary with dynamic hour fields
    result = {
        'indices': [int(i) for i in indices],
        'FIM': final_FIM.tolist(),  # Convert numpy array to list for JSON serialization
        'log10_det': log10_det,
    }

    # Add hour fields dynamically (hour1, hour2, hour3, etc.)
    for idx, hour_val in enumerate(hours, 1):
        result[f'hour{idx}'] = float(hour_val)

    return result


def run_parallel(hours, theta, measurement_error, n_experiments=2, n_cores=None):
    """Parallel computation using multiprocessing.

    Parameters
    ----------
    hours : array-like
        Array of hour values to use for experiments
    theta : dict
        Parameter values for the model
    measurement_error : float
        Measurement error for experiments
    n_experiments : int, optional
        Number of experiments in each combination (2, 3, or 4), default is 2
    n_cores : int, optional
        Number of cores to use, defaults to all available cores

    Returns
    -------
    tuple
        (results, elapsed_time)
    """
    if n_cores is None:
        n_cores = cpu_count()

    # Generate all combinations of n_experiments from hours
    from itertools import combinations_with_replacement

    tasks = []
    for indices in combinations_with_replacement(range(len(hours)), n_experiments):
        hour_values = tuple(hours[i] for i in indices)
        tasks.append((indices, hour_values, theta, measurement_error))

    start_time = time.time()

    # Use multiprocessing Pool to compute in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_fim_combination, tasks)

    elapsed = time.time() - start_time
    return results, elapsed


if __name__ == "__main__":
    # Create hour data with dense sampling around optimal region for verification
    # Optimal from optimization: hour1≈1.90, hour2≈10.0
    hours_p1 = np.linspace(0.1, 1.7, 5)  # Before optimal: coarse
    hours_p2 = np.linspace(1.71, 2.1, 5)  # Around optimal hour1: dense
    hours_p3 = np.linspace(2.2, 9.4, 5)  # Between: coarse
    hours_p4 = np.linspace(9.41, 10, 5)  # Around optimal hour2: dense
    hours = np.concatenate((hours_p1, hours_p2, hours_p3, hours_p4))

    # Set number of experiments (2, 3, or 4)
    n_experiments = 2

    # Calculate number of combinations
    from math import comb

    n_combinations = comb(len(hours) + n_experiments - 1, n_experiments)

    print(f"\nVerification Grid Setup:")
    print(f"  Total points: {len(hours)}")
    print(f"  Range: [{hours[0]:.2f}, {hours[-1]:.2f}]")
    print(f"  Dense around hour1≈1.90: [1.71, 2.10] with {len(hours_p2)} points")
    print(f"  Dense around hour2≈10.0: [9.41, 10.00] with {len(hours_p4)} points")
    print(f"  Number of experiments per combination: {n_experiments}")
    print(f"  Total combinations: {n_combinations}")

    theta = {'asymptote': 15, 'rate_constant': 0.5}
    measurement_error = 0.1

    n_cores = cpu_count()

    print(f"\nComputing FIM for verification using {n_cores} cores...\n")

    # Run parallel computation only
    results_parallel, time_parallel = run_parallel(
        hours, theta, measurement_error, n_experiments, n_cores
    )

    print(f"Computation completed in {time_parallel:.2f} seconds\n")

    # Save results to JSON file
    file_name = f"rooney_biegler_fim_{n_experiments}exp_verification.json"
    DATA_DIR = Path(__file__).parent
    output_file = DATA_DIR / file_name
    print(f"Saving results to {output_file}...")

    output_data = {
        'metadata': {
            'n_experiments': n_experiments,
            'n_hours': len(hours),
            'hour_range': [float(hours[0]), float(hours[-1])],
            'total_computations': len(results_parallel),
            'theta': theta,
            'measurement_error': measurement_error,
            'n_cores': n_cores,
            'computation_time': time_parallel,
            'grid_description': 'Dense sampling around optimal regions (1.71-2.10, 9.41-10), coarse elsewhere',
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

        # Print hour values dynamically
        hour_str = ", ".join(
            [f"Hour{i}={best[f'hour{i}']:.4f}" for i in range(1, n_experiments + 1)]
        )
        print(f"Best design from grid: {hour_str}")
        print(f"                       log10(det) = {best['log10_det']:.4f}")

        if n_experiments == 2:
            print(f"\nOptimization result:   Hour1≈1.900, Hour2≈10.000")
            print(f"                       log10(det) ≈ 6.0280")
        print("=" * 70 + "\n")
