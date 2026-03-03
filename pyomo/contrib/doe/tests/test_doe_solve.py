# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
import json
import logging
import os, os.path
import subprocess
import time
from itertools import combinations, product
from glob import glob
from unittest.mock import patch

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
    matplotlib,
    matplotlib_available,
)

# Set matplotlib backend for non-interactive use (for CI testing purposes)
if matplotlib_available:
    matplotlib.use("Agg")

from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

if scipy_available:
    from pyomo.contrib.doe import DesignOfExperiments
    from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
    from pyomo.contrib.doe.examples.reactor_example import (
        ReactorExperiment as FullReactorExperiment,
        run_reactor_doe,
    )
    from pyomo.contrib.doe.tests.experiment_class_example_flags import (
        RooneyBieglerExperimentBad,
        RooneyBieglerMultiExperiment,
    )
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
        RooneyBieglerExperiment,
    )
from pyomo.contrib.doe.utils import rescale_FIM
from pyomo.contrib.doe.examples.rooney_biegler_doe_example import run_rooney_biegler_doe

import pyomo.environ as pyo

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()
k_aug_available = SolverFactory("k_aug", solver_io="nl", validate=False)


def k_aug_runtime_available():
    """
    Check that k_aug is not only discoverable but also runnable in this
    environment (e.g., no missing dynamic libraries).
    """
    if not k_aug_available.available(False):
        return False
    exe = k_aug_available.executable()
    if not exe:
        return False

    try:
        # Trigger dynamic loader checks; return code may be nonzero for usage,
        # so we inspect output for runtime-linker failures.
        proc = subprocess.run(
            [exe, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return False

    output = (proc.stdout or "") + (proc.stderr or "")
    bad_runtime_markers = ("Library not loaded", "dyld:", "libgfortran")
    if any(marker in output for marker in bad_runtime_markers):
        return False
    return True


currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)
data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}


def get_rooney_biegler_data():
    """Get Rooney-Biegler experiment data for testing."""
    # Create a simple data point for Rooney-Biegler model
    # This must be a pandas Series with 'hour' and 'y' columns
    # Use data from the tested Rooney-Biegler dataset
    data = pd.DataFrame(data=[[5, 15.6]], columns=['hour', 'y'])
    return data.iloc[0]


def get_rooney_biegler_experiment():
    """Get a fresh RooneyBieglerExperiment instance for testing.

    Creates a new experiment instance to ensure test isolation.
    Each test gets its own instance to avoid state sharing.
    """
    return RooneyBieglerExperiment(
        data=get_rooney_biegler_data(),
        theta={'asymptote': 15, 'rate_constant': 0.5},
        measure_error=0.1,
    )


def get_FIM_Q_L(doe_obj=None):
    """
    Helper function to retrieve results to compare.

    """
    model = doe_obj.model

    n_param = doe_obj.n_parameters
    n_y = doe_obj.n_experiment_outputs

    FIM_vals = [
        pyo.value(model.fim[i, j])
        for i in model.parameter_names
        for j in model.parameter_names
    ]
    if hasattr(model, "L"):
        L_vals = [
            pyo.value(model.L[i, j])
            for i in model.parameter_names
            for j in model.parameter_names
        ]
    else:
        L_vals = [[0] * n_param] * n_param
    Q_vals = [
        pyo.value(model.sensitivity_jacobian[i, j])
        for i in model.output_names
        for j in model.parameter_names
    ]
    sigma_inv = [
        1 / v**2 for k, v in model.fd_scenario_blocks[0].measurement_error.items()
    ]
    param_vals = np.array(
        [[v for k, v in model.fd_scenario_blocks[0].unknown_parameters.items()]]
    )

    FIM_vals_np = np.array(FIM_vals).reshape((n_param, n_param))

    for i in range(n_param):
        for j in range(n_param):
            if j < i:
                FIM_vals_np[j, i] = FIM_vals_np[i, j]

    L_vals_np = np.array(L_vals).reshape((n_param, n_param))
    Q_vals_np = np.array(Q_vals).reshape((n_y, n_param))

    sigma_inv_np = np.zeros((n_y, n_y))

    for ind, v in enumerate(sigma_inv):
        sigma_inv_np[ind, ind] = v

    return FIM_vals_np, Q_vals_np, L_vals_np, sigma_inv_np


def get_standard_args(experiment, fd_method, obj_used):
    args = {}
    args['experiment_list'] = None if experiment is None else [experiment]
    args['fd_formula'] = fd_method
    args['step'] = 1e-3
    args['objective_option'] = obj_used
    args['scale_constant_value'] = 1
    args['scale_nominal_param_value'] = True
    args['prior_FIM'] = None
    args['jac_initial'] = None
    args['fim_initial'] = None
    args['L_diagonal_lower_bound'] = 1e-7
    # Make solver object with good linear subroutines
    solver = SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["max_iter"] = 3000
    args['solver'] = solver
    args['tee'] = False
    args['get_labeled_model_args'] = None
    args['_Cholesky_option'] = True
    args['_only_compute_fim_lower'] = True
    return args


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
class TestRooneyBieglerExampleSolving(unittest.TestCase):
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_fd_central_solve(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        # Use RooneyBiegler for algorithm validation (faster)
        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        # assert model solves
        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_fd_forward_solve(self):
        fd_method = "forward"
        obj_used = "zero"

        # Use RooneyBiegler for algorithm validation (faster)
        # Use hour=7 for better FIM conditioning with zero objective
        data_point = pd.DataFrame({'hour': [7.0], 'y': [19.8]}).iloc[0]

        experiment = RooneyBieglerExperiment(
            data=data_point,
            theta={'asymptote': 15, 'rate_constant': 0.5},
            measure_error=0.1,
        )

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        # Add prior FIM to avoid singularity with zero objective
        # This follows the pattern in rooney_biegler_doe_example.py
        doe_obj_prior = DesignOfExperiments(**DoE_args)
        prior_FIM = doe_obj_prior.compute_FIM()
        DoE_args['prior_FIM'] = prior_FIM

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Note: When using prior_FIM, the relationship FIM = Q.T @ sigma_inv @ Q + prior_FIM
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q + prior_FIM)))

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_fd_backward_solve(self):
        fd_method = "backward"
        obj_used = "pseudo_trace"

        # Use RooneyBiegler for algorithm validation (faster)
        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_obj_det_solve(self):
        fd_method = "central"
        obj_used = "determinant"

        # Use RooneyBiegler for algorithm validation (faster)
        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["scale_nominal_param_value"] = (
            False  # Vanilla determinant solve needs this
        )
        DoE_args["_Cholesky_option"] = False
        DoE_args["_only_compute_fim_lower"] = False

        doe_obj = DesignOfExperiments(**DoE_args)

        # Increase numerical performance by adding a prior
        prior_FIM = doe_obj.compute_FIM()
        doe_obj.prior_FIM = prior_FIM

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        expected_design = 9.999213890476453
        actual_design = doe_obj.results["Experiment Design"][0]
        self.assertAlmostEqual(actual_design, expected_design, places=3)

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_obj_cholesky_solve(self):
        fd_method = "central"
        obj_used = "determinant"

        # Use RooneyBiegler for algorithm validation (faster)
        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        # Add prior FIM for better numerical conditioning
        # This follows the pattern in rooney_biegler_doe_example.py
        doe_obj_prior = DesignOfExperiments(**DoE_args)
        prior_FIM = doe_obj_prior.compute_FIM()
        DoE_args['prior_FIM'] = prior_FIM

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Cholesky is used, there is comparison for FIM and L.T @ L
        self.assertTrue(np.all(np.isclose(FIM, L @ L.T)))

        # Note: When using prior_FIM, the relationship FIM = Q.T @ sigma_inv @ Q + prior_FIM
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q + prior_FIM)))

    def DISABLE_test_reactor_obj_cholesky_solve_bad_prior(self):
        # [10/2025] This test has been disabled because it frequently
        # (and randomly) returns "infeasible" when run on Windows.
        from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS

        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        # Specify a prior that is slightly negative definite
        # Because it is less than the tolerance, it should be
        # adjusted to be positive definite
        # No error should be thrown
        DoE_args["prior_FIM"] = -(_SMALL_TOLERANCE_DEFINITENESS / 100) * np.eye(4)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Cholesky is used, there is comparison for FIM and L.T @ L
        self.assertTrue(np.all(np.isclose(FIM, L @ L.T)))

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    # This test ensure that compute FIM runs without error using the
    # `sequential` option with central finite differences
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_compute_FIM_seq_centr(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        # Use RooneyBiegler for algorithm validation (faster)
        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        expected_FIM = np.array(
            [[18957.7788694, 4238.27606876], [4238.27606876, 947.52577076]]
        )

        self.assertTrue(
            np.all(np.isclose(doe_obj.compute_FIM(method="sequential"), expected_FIM))
        )

    # This test ensure that compute FIM runs without error using the
    # `sequential` option with forward finite differences
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_compute_FIM_seq_forward(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        # Use RooneyBiegler for algorithm validation (faster)
        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

    # This test ensure that compute FIM runs without error using the
    # `kaug` option. kaug computes the FIM directly so no finite difference
    # scheme is needed.
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    @unittest.skipIf(
        not k_aug_runtime_available(),
        "The 'k_aug' command is not available or not runnable in this environment",
    )
    def test_compute_FIM_kaug(self):
        fd_method = "forward"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        expected_FIM = np.array(
            [[18957.7788694, 4238.27606876], [4238.27606876, 947.52577076]]
        )

        self.assertTrue(
            np.all(np.isclose(doe_obj.compute_FIM(method="kaug"), expected_FIM))
        )

    # This test ensure that compute FIM runs without error using the
    # `sequential` option with backward finite differences
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_compute_FIM_seq_backward(self):
        fd_method = "backward"
        obj_used = "pseudo_trace"

        # Use RooneyBiegler for algorithm validation (faster)
        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_grid_search(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        # Reduce grid from 3x3 to 2x2 for performance
        design_ranges = {"CA[0]": [1, 5, 2], "T[0]": [300, 700, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        # Check to make sure the lengths of the inputs
        # in results object are indeed correct
        CA_vals = doe_obj.fim_factorial_results["CA[0]"]
        T_vals = doe_obj.fim_factorial_results["T[0]"]

        # assert length is correct (2x2 = 4 evaluations)
        self.assertTrue((len(CA_vals) == 4) and (len(T_vals) == 4))
        self.assertTrue((len(set(CA_vals)) == 2) and (len(set(T_vals)) == 2))

        # assert unique values are correct
        self.assertTrue(
            (set(CA_vals).issuperset(set([1, 5])))
            and (set(T_vals).issuperset(set([300, 700])))
        )

    def test_rescale_FIM(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        # With parameter scaling
        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        # Without parameter scaling
        DoE_args2 = get_standard_args(experiment, fd_method, obj_used)
        DoE_args2["scale_nominal_param_value"] = False

        doe_obj2 = DesignOfExperiments(**DoE_args2)
        # Run both problems
        doe_obj.run_doe()
        doe_obj2.run_doe()

        # Extract FIM values
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)
        FIM2, Q2, L2, sigma_inv2 = get_FIM_Q_L(doe_obj=doe_obj2)

        # Get rescaled FIM from the scaled version
        param_vals = np.array(
            [
                [
                    v
                    for k, v in doe_obj.model.fd_scenario_blocks[
                        0
                    ].unknown_parameters.items()
                ]
            ]
        )

        resc_FIM = rescale_FIM(FIM, param_vals)

        # Compare scaled and rescaled values
        self.assertTrue(np.all(np.isclose(FIM2, resc_FIM)))

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_solve_bad_model(self):
        fd_method = "central"
        obj_used = "determinant"

        # Use RooneyBiegler bad example (faster than reactor bad example)
        experiment = RooneyBieglerExperimentBad(
            data=get_rooney_biegler_data(),
            theta={'asymptote': 15, 'rate_constant': 0.5},
            measure_error=0.1,
        )

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model from experiment did not solve appropriately. "
            "Make sure the model is well-posed.",
        ):
            doe_obj.run_doe()

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_grid_search_bad_model(self):
        fd_method = "central"
        obj_used = "determinant"

        # Use RooneyBiegler bad example (faster than reactor bad example)
        experiment = RooneyBieglerExperimentBad(
            data=get_rooney_biegler_data(),
            theta={'asymptote': 15, 'rate_constant': 0.5},
            measure_error=0.1,
        )

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["logger_level"] = logging.ERROR

        doe_obj = DesignOfExperiments(**DoE_args)

        # Use simpler design ranges for RooneyBiegler
        design_ranges = {"hour": [1, 5, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        # Check to make sure the lengths of the inputs in results object are indeed correct
        hour_vals = doe_obj.fim_factorial_results["hour"]

        # assert length is correct
        self.assertTrue(len(hour_vals) == 2)
        self.assertTrue(len(set(hour_vals)) == 2)

        # assert unique values are correct
        self.assertTrue(set(hour_vals).issuperset(set([1, 5])))


@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestDoe(unittest.TestCase):
    def test_doe_full_factorial(self):
        log10_D_opt_expected = [
            11.77343778527225,
            13.137792359064383,
            13.182167857699808,
            14.54652243150573,
        ]

        log10_A_opt_expected = [
            5.59357268009304,
            5.613318615148643,
            5.945755198204368,
            5.965501133259909,
        ]

        log10_E_opt_expected = [
            0.27981268741620413,
            1.3086595026369012,
            0.6319952055040333,
            1.6608420207466377,
        ]

        log10_ME_opt_expected = [
            5.221185311075697,
            4.244741560076784,
            5.221185311062606,
            4.244741560083524,
        ]

        eigval_min_expected = [
            1.9046390638130666,
            20.354456134677426,
            4.285437893696232,
            45.797526302234304,
        ]

        eigval_max_expected = [
            316955.2855492114,
            357602.92523637977,
            713149.3924857995,
            804606.58178139165,
        ]

        det_FIM_expected = [
            593523317093.4525,
            13733851875566.766,
            15211353450350.424,
            351983602166961.56,
        ]

        trace_FIM_expected = [
            392258.78617108597,
            410505.1549241871,
            882582.2688850109,
            923636.598578955,
        ]
        ff = run_reactor_doe(
            n_points_for_design=2,
            compute_FIM_full_factorial=False,
            plot_factorial_results=False,
            run_optimal_doe=False,
        )
        ff.compute_FIM_full_factorial(
            design_ranges={"CA[0]": [1, 1.5, 2], "T[0]": [350, 400, 2]}
        )

        ff_results = ff.fim_factorial_results

        self.assertStructuredAlmostEqual(
            ff_results["log10 D-opt"], log10_D_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["log10 pseudo A-opt"], log10_A_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["log10 E-opt"], log10_E_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["log10 ME-opt"], log10_ME_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["eigval_min"], eigval_min_expected, abstol=1e-4
        )
        # abstol of 1e-4 removed for the following values as
        # their non-log values are large (e.g., >1e10)
        self.assertStructuredAlmostEqual(ff_results["eigval_max"], eigval_max_expected)
        self.assertStructuredAlmostEqual(ff_results["det_FIM"], det_FIM_expected)
        self.assertStructuredAlmostEqual(ff_results["trace_FIM"], trace_FIM_expected)

    @unittest.skipUnless(pandas_available, "test requires pandas")
    def test_doe_A_optimality(self):
        A_opt_value_expected = -2.2364242059539663
        A_opt_design_value_expected = 9.999955457176451

        A_opt_res = run_rooney_biegler_doe(optimization_objective="trace")
        A_opt_value = A_opt_res["optimization"]["value"]
        A_opt_design_value = A_opt_res["optimization"]["design"][0]

        self.assertAlmostEqual(A_opt_value, A_opt_value_expected, places=2)
        # print("A optimal design value:", A_opt_design_value)
        self.assertAlmostEqual(
            A_opt_design_value, A_opt_design_value_expected, places=2
        )


class TestRooneyBieglerExample(unittest.TestCase):
    @unittest.skipUnless(pandas_available, "test requires pandas")
    @unittest.skipUnless(ipopt_available, "test requires ipopt")
    def test_rooney_biegler_doe_example(self):
        """
        Tests the Design of Experiments (DoE) functionality, including
        plotting logic when matplotlib is available, without displaying GUI windows.
        """
        file_prefix = "rooney_biegler"

        # Cleanup function for generated files
        def cleanup_file():
            generated_files = glob(f"{file_prefix}_*.png")
            for f in generated_files:
                try:
                    os.remove(f)
                except OSError:
                    pass

        self.addCleanup(cleanup_file)

        # Run with draw_factorial_figure conditional on matplotlib availability
        # Test D-optimality
        results_D = run_rooney_biegler_doe(
            optimization_objective="determinant",
            compute_FIM_full_factorial=True,
            draw_factorial_figure=matplotlib_available,
            design_range={'hour': [0, 10, 3]},
            tee=False,
        )

        # Test A-optimality
        results_A = run_rooney_biegler_doe(
            optimization_objective="trace",
            compute_FIM_full_factorial=False,
            draw_factorial_figure=False,
            design_range={'hour': [0, 10, 3]},
            tee=False,
        )

        # Assertions for Numerical Results
        self.assertEqual("determinant", results_D["optimization"]["objective_type"])
        self.assertEqual("trace", results_A["optimization"]["objective_type"])

        # Test D-optimality optimization results
        D_opt_value_expected = 6.864794717802814
        D_opt_design_value_expected = 10.0  # approximately 9.999999472662282

        D_opt_value = results_D["optimization"]["value"]
        D_opt_design_value = results_D["optimization"]["design"][0]

        self.assertAlmostEqual(D_opt_value, D_opt_value_expected, places=4)
        self.assertAlmostEqual(
            D_opt_design_value, D_opt_design_value_expected, places=4
        )

        # Test A-optimality optimization results
        A_opt_value_expected = -2.236424205953928
        A_opt_design_value_expected = 10.0  # approximately 9.999955457176451

        A_opt_value = results_A["optimization"]["value"]
        A_opt_design_value = results_A["optimization"]["design"][0]

        self.assertAlmostEqual(A_opt_value, A_opt_value_expected, places=4)
        self.assertAlmostEqual(
            A_opt_design_value, A_opt_design_value_expected, places=4
        )

        # Assertions for Full Factorial Results
        self.assertIn("results_dict", results_D)
        results_dict = results_D["results_dict"]
        self.assertIsInstance(results_dict, dict)
        self.assertGreater(len(results_dict), 0, "results_dict should not be empty")

        # Expected values for design_range={'hour': [0, 10, 3]}
        # These are the 3 data points from the full factorial grid
        expected_log10_D_opt = [6.583798747893548, 6.691228337572129, 6.864794726228617]
        expected_log10_A_opt = [
            -1.9574859220185146,
            -2.0268526846104975,
            -2.236424954559946,
        ]
        expected_log10_pseudo_A_opt = [
            4.62631282587503,
            4.6643756529616285,
            4.628369771668666,
        ]
        expected_log10_E_opt = [
            1.9584199467177335,
            2.0278567624056967,
            2.2381970919918426,
        ]
        expected_log10_ME_opt = [
            2.666958854458125,
            2.6355148127607713,
            2.3884005422449364,
        ]

        # Verify structure and values using assertStructuredAlmostEqual
        self.assertStructuredAlmostEqual(
            results_dict["log10 D-opt"], expected_log10_D_opt, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            results_dict["log10 A-opt"], expected_log10_A_opt, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            results_dict["log10 pseudo A-opt"], expected_log10_pseudo_A_opt, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            results_dict["log10 E-opt"], expected_log10_E_opt, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            results_dict["log10 ME-opt"], expected_log10_ME_opt, abstol=1e-4
        )

        # Plot-related assertions only when matplotlib is available
        if matplotlib_available:
            # Check that draw_factorial_figure actually created the file
            expected_d_plot = f"{file_prefix}_D_opt.png"
            self.assertTrue(
                os.path.exists(expected_d_plot),
                f"Expected plot file '{expected_d_plot}' was not created.",
            )


@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not pandas_available, "Pandas is not available")
@unittest.skipIf(not matplotlib_available, "Matplotlib is not available")
class TestDoEFactorialFigure(unittest.TestCase):
    def test_doe_1D_plotting_function(self):
        # For 1D plotting we will use the Rooney-Biegler example in parmest/examples
        plt = matplotlib.pyplot
        """
        Test that the plotting function executes without error and
        creates a matplotlib figure. We do NOT test visual correctness.
        """

        # File prefix for saved plots
        # Define prefixes for the two runs
        prefix_linear = "rooney_linear"
        prefix_log = "rooney_log"

        # Clean up any existing plot files from test runs
        def cleanup_files():
            files_to_remove = glob("rooney_*.png")
            for f in files_to_remove:
                try:
                    os.remove(f)
                except OSError:
                    pass
            plt.close('all')

        self.addCleanup(cleanup_files)

        fd_method = "central"
        obj_used = "trace"

        experiment = run_rooney_biegler_doe()["experiment"]

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM_full_factorial(design_ranges={'hour': [0, 10, 1]})

        # Call the plotting function for linear scale
        doe_obj.draw_factorial_figure(
            sensitivity_design_variables=['hour'],
            fixed_design_variables={},
            log_scale=False,
            figure_file_name=prefix_linear,
        )

        # Call the plotting function for log scale
        doe_obj.draw_factorial_figure(
            sensitivity_design_variables=['hour'],
            fixed_design_variables={},
            log_scale=True,
            figure_file_name=prefix_log,
        )

        # Verify that the linear scale plots were also created
        # Check that we found exactly 5 files (A, D, E, ME, pseudo_A)
        expected_plot_linear = glob(f"{prefix_linear}*.png")
        self.assertEqual(
            len(expected_plot_linear),
            5,
            f"Expected 5 plot files, but found {len(expected_plot_linear)}. Files found: {expected_plot_linear}",
        )

        # Verify that the log scale plots were also created
        expected_plot_log = glob(f"{prefix_log}*.png")
        self.assertEqual(
            len(expected_plot_log),
            5,
            f"Expected 5 plot files, but found {len(expected_plot_log)}. Files found: {expected_plot_log}",
        )

    def test_doe_2D_plotting_function(self):
        # For 2D plotting we will use the Rooney-Biegler example in doe/examples
        plt = matplotlib.pyplot

        # File prefix for saved plots
        prefix_linear = "reactor_linear"
        prefix_log = "reactor_log"

        # Clean up any existing plot files from test runs
        def cleanup_files():
            files_to_remove = glob("reactor_*.png")
            for f in files_to_remove:
                try:
                    os.remove(f)
                except OSError:
                    pass
            plt.close('all')

        self.addCleanup(cleanup_files)

        # Run the reactor example
        run_reactor_doe(
            n_points_for_design=1,
            compute_FIM_full_factorial=True,
            plot_factorial_results=True,
            figure_file_name=prefix_linear,
            log_scale=False,
            run_optimal_doe=False,
        )

        # Verify that the linear scale plots were also created
        # Check that we found exactly 5 files (A, D, E, ME, pseudo_A)
        expected_plot_linear = glob(f"{prefix_linear}*.png")
        self.assertTrue(
            len(expected_plot_linear) == 5,
            f"Expected 5 plot files, but found {len(expected_plot_linear)}. Files found: {expected_plot_linear}",
        )

        # Run the reactor example with log scale
        run_reactor_doe(
            n_points_for_design=1,
            compute_FIM_full_factorial=True,
            plot_factorial_results=True,
            figure_file_name=prefix_log,
            log_scale=True,
            run_optimal_doe=False,
        )

        # Verify that the log scale plots were also created
        # Check that we found exactly 5 files (A, D, E, ME, pseudo_A)
        expected_plot_log = glob(f"{prefix_log}*.png")
        self.assertTrue(
            len(expected_plot_log) == 5,
            f"Expected 5 plot files, but found {len(expected_plot_log)}. Files found: {expected_plot_log}",
        )


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
class TestOptimizeExperimentsAlgorithm(unittest.TestCase):
    def _make_template_doe(self, objective_option="pseudo_trace"):
        exp = RooneyBieglerMultiExperiment(hour=2.0, y=10.0)
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        return DesignOfExperiments(
            experiment_list=[exp],
            objective_option=objective_option,
            step=1e-2,
            solver=solver,
        )

    def _build_template_model_for_multi_experiment(self, doe_obj, n_exp):
        doe_obj.model.param_scenario_blocks = pyo.Block(range(1))
        doe_obj.model.param_scenario_blocks[0].exp_blocks = pyo.Block(range(n_exp))
        for k in range(n_exp):
            doe_obj.create_doe_model(
                model=doe_obj.model.param_scenario_blocks[0].exp_blocks[k],
                experiment_index=0,
                _for_multi_experiment=True,
            )

    def test_evaluate_objective_from_fim_numerical_values(self):
        fim = np.array([[4.0, 1.0], [1.0, 3.0]])

        doe_det = self._make_template_doe("determinant")
        self.assertAlmostEqual(
            doe_det._evaluate_objective_from_fim(fim), np.linalg.det(fim), places=10
        )

        doe_ptr = self._make_template_doe("pseudo_trace")
        self.assertAlmostEqual(
            doe_ptr._evaluate_objective_from_fim(fim), np.trace(fim), places=10
        )

        doe_tr = self._make_template_doe("trace")
        self.assertAlmostEqual(
            doe_tr._evaluate_objective_from_fim(fim),
            np.trace(np.linalg.inv(fim)),
            places=10,
        )

    def test_evaluate_objective_from_fim_fallback_paths(self):
        singular_fim = np.zeros((2, 2))

        doe_trace = self._make_template_doe("trace")
        self.assertEqual(doe_trace._evaluate_objective_from_fim(singular_fim), np.inf)

        doe_zero = self._make_template_doe("zero")
        self.assertEqual(doe_zero._evaluate_objective_from_fim(singular_fim), 0.0)

    def test_compute_fim_at_point_no_prior_restores_prior(self):
        doe_no_prior = self._make_template_doe("pseudo_trace")
        doe_no_prior.prior_FIM = np.zeros((2, 2))
        expected = doe_no_prior.compute_FIM(method="sequential")

        doe = self._make_template_doe("pseudo_trace")
        saved_prior = np.array([[7.0, 0.0], [0.0, 5.0]])
        doe.prior_FIM = saved_prior
        got = doe._compute_fim_at_point_no_prior(experiment_index=0, input_values=[2.0])

        self.assertTrue(np.allclose(got, expected, atol=1e-8))
        self.assertIs(doe.prior_FIM, saved_prior)
        self.assertTrue(np.allclose(doe.prior_FIM, saved_prior, atol=1e-12))

    def test_compute_fim_at_point_no_prior_exception_fallback_zero(self):
        doe = self._make_template_doe("pseudo_trace")
        saved_prior = np.array([[3.0, 0.0], [0.0, 4.0]])
        doe.prior_FIM = saved_prior

        with patch.object(doe, "_sequential_FIM", side_effect=RuntimeError("boom")):
            with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as log_cm:
                got = doe._compute_fim_at_point_no_prior(
                    experiment_index=0, input_values=[2.0]
                )

        self.assertTrue(np.allclose(got, np.zeros((2, 2))))
        self.assertIs(doe.prior_FIM, saved_prior)
        self.assertTrue(
            any("Using zero matrix as fallback" in msg for msg in log_cm.output)
        )

    def test_optimize_experiments_cholesky_jitter_branch(self):
        doe = self._make_template_doe("determinant")

        original_solve = doe.solver.solve
        solve_count = {"n": 0}

        class _MockSolverInfo:
            status = "ok"
            termination_condition = "optimal"
            message = "mock-solve"

        class _MockResults:
            solver = _MockSolverInfo()

        def _solve_first_real_then_mock(*args, **kwargs):
            solve_count["n"] += 1
            if solve_count["n"] == 1:
                return original_solve(*args, **kwargs)
            return _MockResults()

        with patch(
            "pyomo.contrib.doe.doe.np.linalg.eigvals",
            return_value=np.array([-1.0, 1.0]),
        ) as eigvals_mock:
            with patch.object(
                doe.solver, "solve", side_effect=_solve_first_real_then_mock
            ):
                doe.optimize_experiments(n_exp=1)

        self.assertEqual(doe.results["Solver Status"], "ok")
        self.assertGreaterEqual(eigvals_mock.call_count, 1)

    def test_optimize_experiments_lhs_matches_bruteforce_combo(self):
        n_exp = 2
        lhs_n_samples = 3
        lhs_seed = 17

        # Build expected best combo using the same helper path and objective scoring.
        expected_obj = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(expected_obj, n_exp=n_exp)

        first_exp_block = expected_obj.model.param_scenario_blocks[0].exp_blocks[0]
        exp_input_vars = expected_obj._get_experiment_input_vars(first_exp_block)
        lb_vals = np.array([v.lb for v in exp_input_vars])
        ub_vals = np.array([v.ub for v in exp_input_vars])

        rng = np.random.default_rng(lhs_seed)
        from scipy.stats.qmc import LatinHypercube

        per_dim_samples = []
        for i in range(len(exp_input_vars)):
            dim_seed = int(rng.integers(0, 2**31))
            sampler = LatinHypercube(d=1, seed=dim_seed)
            s_unit = sampler.random(n=lhs_n_samples).flatten()
            s_scaled = lb_vals[i] + s_unit * (ub_vals[i] - lb_vals[i])
            per_dim_samples.append(s_scaled.tolist())

        candidate_points = list(product(*per_dim_samples))
        candidate_fims = [
            expected_obj._compute_fim_at_point_no_prior(0, list(pt))
            for pt in candidate_points
        ]

        best_combo = None
        best_obj = -np.inf
        for combo in combinations(range(len(candidate_points)), n_exp):
            fim_total = sum((candidate_fims[idx] for idx in combo), np.zeros((2, 2)))
            obj_val = expected_obj._evaluate_objective_from_fim(fim_total)
            if obj_val > best_obj:
                best_obj = obj_val
                best_combo = combo

        expected_points = [list(candidate_points[i]) for i in best_combo]

        # Run full optimization with LHS initialization and compare chosen points.
        doe = self._make_template_doe("pseudo_trace")
        doe.optimize_experiments(
            n_exp=n_exp,
            initialization_method="lhs",
            init_n_samples=lhs_n_samples,
            init_seed=lhs_seed,
        )

        actual_points = doe.results["LHS Best Initial Points"]
        actual_points_norm = sorted(tuple(np.round(p, 8)) for p in actual_points)
        expected_points_norm = sorted(tuple(np.round(p, 8)) for p in expected_points)
        self.assertEqual(actual_points_norm, expected_points_norm)
        self.assertEqual(doe.results["Initialization Method"], "lhs")

        # Numerical consistency of aggregated FIM in result payload.
        scenario = doe.results["Scenarios"][0]
        total_fim = np.array(scenario["Total FIM"])
        prior = np.array(doe.results["Prior FIM"])
        exp_fim_sum = sum(
            (np.array(exp_data["FIM"]) for exp_data in scenario["Experiments"]),
            np.zeros_like(total_fim),
        )
        self.assertTrue(np.allclose(total_fim, exp_fim_sum + prior, atol=1e-6))

    def test_optimize_experiments_is_reentrant_on_same_object(self):
        doe = self._make_template_doe("pseudo_trace")
        doe.optimize_experiments(n_exp=1)
        first_design = doe.results["Scenarios"][0]["Experiments"][0][
            "Experiment Design"
        ]
        first_build_time = doe.results["timing"]["build_s"]

        doe.optimize_experiments(n_exp=1)
        second_design = doe.results["Scenarios"][0]["Experiments"][0][
            "Experiment Design"
        ]

        self.assertEqual(len(first_design), len(second_design))
        self.assertIn("timing", doe.results)
        self.assertGreater(doe.results["timing"]["build_s"], 0.0)
        self.assertGreaterEqual(first_build_time, 0.0)
        self.assertEqual(len(list(doe.model.param_scenario_blocks.keys())), 1)

    def test_lhs_combo_parallel_matches_serial(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        # Use deterministic synthetic FIMs to isolate combo scorer behavior.
        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, 2.0 * x + 1.0]])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_serial, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=4, lhs_seed=123, n_exp=2, lhs_combo_parallel=False
            )

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_parallel, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=4,
                lhs_seed=123,
                n_exp=2,
                lhs_combo_parallel=True,
                lhs_n_workers=2,
                lhs_combo_chunk_size=2,
                lhs_combo_parallel_threshold=1,
            )

        serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
        parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
        self.assertEqual(serial_norm, parallel_norm)

    def test_lhs_parallel_fim_eval_matches_serial(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        # Patch at class level so both serial path (self) and parallel worker
        # DOE objects use the same deterministic synthetic FIM mapping.
        def _fake_fim(self_obj, experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 0.5, 0.0], [0.0, 3.0 * x + 0.5]])

        with patch.object(
            DesignOfExperiments,
            "_compute_fim_at_point_no_prior",
            autospec=True,
            side_effect=_fake_fim,
        ):
            points_serial, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=4, lhs_seed=321, n_exp=2, lhs_parallel=False
            )

            points_parallel, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=4,
                lhs_seed=321,
                n_exp=2,
                lhs_parallel=True,
                lhs_n_workers=2,
            )

        serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
        parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
        self.assertEqual(serial_norm, parallel_norm)

    def test_lhs_parallel_fim_eval_real_path_smoke(self):
        doe_serial = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe_serial, n_exp=2)
        points_serial, _ = doe_serial._lhs_initialize_experiments(
            lhs_n_samples=3,
            lhs_seed=9,
            n_exp=2,
            lhs_parallel=False,
            lhs_combo_parallel=False,
        )

        doe_parallel = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe_parallel, n_exp=2)
        points_parallel, _ = doe_parallel._lhs_initialize_experiments(
            lhs_n_samples=3,
            lhs_seed=9,
            n_exp=2,
            lhs_parallel=True,
            lhs_n_workers=2,
            lhs_combo_parallel=False,
        )

        serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
        parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
        self.assertEqual(serial_norm, parallel_norm)

    def test_lhs_parallel_solver_without_name_uses_default_worker_solver(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)
        doe.solver = object()

        def _fake_fim(self_obj, experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 0.5, 0.0], [0.0, x + 1.0]])

        with patch.object(
            DesignOfExperiments,
            "_compute_fim_at_point_no_prior",
            autospec=True,
            side_effect=_fake_fim,
        ):
            with self.assertLogs("pyomo.contrib.doe.doe", level="DEBUG") as log_cm:
                points, _ = doe._lhs_initialize_experiments(
                    lhs_n_samples=4,
                    lhs_seed=14,
                    n_exp=2,
                    lhs_parallel=True,
                    lhs_n_workers=2,
                    lhs_combo_parallel=False,
                )

        self.assertEqual(len(points), 2)
        self.assertTrue(
            any("solver has no 'name' attribute" in msg for msg in log_cm.output)
        )

    def test_lhs_parallel_solver_factory_failure_uses_default_worker_solver(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        class _NamedSolver:
            name = "definitely_missing_solver"
            options = {}

        doe.solver = _NamedSolver()

        def _fake_fim(self_obj, experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 0.5, 0.0], [0.0, x + 1.0]])

        with patch("pyomo.contrib.doe.doe.pyo.SolverFactory", return_value=None):
            with patch.object(
                DesignOfExperiments,
                "_compute_fim_at_point_no_prior",
                autospec=True,
                side_effect=_fake_fim,
            ):
                with self.assertLogs("pyomo.contrib.doe.doe", level="DEBUG") as log_cm:
                    points, _ = doe._lhs_initialize_experiments(
                        lhs_n_samples=4,
                        lhs_seed=18,
                        n_exp=2,
                        lhs_parallel=True,
                        lhs_n_workers=2,
                        lhs_combo_parallel=False,
                    )

        self.assertEqual(len(points), 2)
        self.assertTrue(
            any(
                "could not construct solver 'definitely_missing_solver'" in msg
                for msg in log_cm.output
            )
        )

    def test_lhs_parallel_worker_exception_uses_zero_fim_fallback(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        with patch.object(
            DesignOfExperiments,
            "_compute_fim_at_point_no_prior",
            side_effect=RuntimeError("worker boom"),
        ):
            with self.assertLogs("pyomo.contrib.doe.doe", level="ERROR") as log_cm:
                points, diag = doe._lhs_initialize_experiments(
                    lhs_n_samples=4,
                    lhs_seed=22,
                    n_exp=2,
                    lhs_parallel=True,
                    lhs_n_workers=2,
                    lhs_combo_parallel=False,
                )

        self.assertEqual(len(points), 2)
        self.assertTrue(diag["timed_out"] is False)
        self.assertTrue(
            any(
                "Using zero FIM for this candidate and continuing." in msg
                for msg in log_cm.output
            )
        )

    def test_lhs_parallel_candidate_timeout_cancels_pending(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _slow_fim(self_obj, experiment_index, input_values):
            time.sleep(0.05)
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 2.0]])

        with patch.object(
            DesignOfExperiments,
            "_compute_fim_at_point_no_prior",
            autospec=True,
            side_effect=_slow_fim,
        ):
            points, diag = doe._lhs_initialize_experiments(
                lhs_n_samples=8,
                lhs_seed=5,
                n_exp=2,
                lhs_parallel=True,
                lhs_n_workers=2,
                lhs_combo_parallel=False,
                lhs_max_wall_clock_time=0.001,
            )

        self.assertEqual(len(points), 2)
        self.assertTrue(diag["timed_out"])

    def test_lhs_no_candidate_fim_scored_uses_zero_fallback(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=1)

        with patch.object(doe, "_compute_fim_at_point_no_prior", return_value=None):
            points, diag = doe._lhs_initialize_experiments(
                lhs_n_samples=3,
                lhs_seed=8,
                n_exp=1,
                lhs_parallel=False,
                lhs_combo_parallel=False,
            )

        self.assertEqual(len(points), 1)
        self.assertTrue(diag["timed_out"])

    def test_lhs_candidate_subset_padding_to_n_exp(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=3)
        call_state = {"n": 0}

        def _fim_with_one_slow_call(experiment_index, input_values):
            call_state["n"] += 1
            x = float(input_values[0])
            if call_state["n"] == 2:
                time.sleep(0.1)
            return np.array([[x + 1.0, 0.0], [0.0, x + 2.0]])

        with patch.object(
            doe, "_compute_fim_at_point_no_prior", side_effect=_fim_with_one_slow_call
        ):
            points, diag = doe._lhs_initialize_experiments(
                lhs_n_samples=5,
                lhs_seed=16,
                n_exp=3,
                lhs_parallel=False,
                lhs_combo_parallel=False,
                lhs_max_wall_clock_time=0.05,
            )

        self.assertEqual(len(points), 3)
        self.assertTrue(diag["timed_out"])
        self.assertLess(diag["n_candidates"], 5)

    def test_lhs_combo_parallel_chunk_boundary_matches_serial(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 2.0, 0.0], [0.0, x + 1.0]])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_serial, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5, lhs_seed=99, n_exp=2, lhs_combo_parallel=False
            )

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_parallel, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5,
                lhs_seed=99,
                n_exp=2,
                lhs_combo_parallel=True,
                lhs_n_workers=2,
                lhs_combo_chunk_size=3,  # C(5,2)=10, non-divisible chunking
                lhs_combo_parallel_threshold=1,
            )

        serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
        parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
        self.assertEqual(serial_norm, parallel_norm)

    def test_lhs_combo_parallel_warning_reports_single_worker_reason(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as log_cm:
                points, _ = doe._lhs_initialize_experiments(
                    lhs_n_samples=4,
                    lhs_seed=27,
                    n_exp=2,
                    lhs_combo_parallel=True,
                    lhs_n_workers=1,
                    lhs_combo_parallel_threshold=1,
                )

        self.assertEqual(len(points), 2)
        self.assertTrue(any("resolved_workers=1 <= 1" in msg for msg in log_cm.output))

    def test_lhs_combo_parallel_skips_empty_local_combo(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            with patch.object(
                DesignOfExperiments,
                "_evaluate_objective_for_option",
                return_value=float("nan"),
            ):
                points, _ = doe._lhs_initialize_experiments(
                    lhs_n_samples=4,
                    lhs_seed=33,
                    n_exp=2,
                    lhs_combo_parallel=True,
                    lhs_n_workers=2,
                    lhs_combo_chunk_size=2,
                    lhs_combo_parallel_threshold=1,
                )

        self.assertEqual(len(points), 2)

    def test_lhs_combo_scoring_timeout_returns_best_so_far(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        def _slow_obj(fim, objective_option):
            time.sleep(0.003)
            return float(np.trace(fim))

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            with patch.object(
                DesignOfExperiments,
                "_evaluate_objective_for_option",
                side_effect=_slow_obj,
            ):
                with self.assertLogs(
                    "pyomo.contrib.doe.doe", level="WARNING"
                ) as log_cm:
                    points, diag = doe._lhs_initialize_experiments(
                        lhs_n_samples=6,
                        lhs_seed=7,
                        n_exp=2,
                        lhs_combo_parallel=False,
                        lhs_max_wall_clock_time=0.001,
                    )

        self.assertEqual(len(points), 2)
        self.assertTrue(any("time budget" in msg for msg in log_cm.output))
        self.assertTrue(diag["timed_out"])

    def test_lhs_combo_scoring_parallel_timeout_returns_best_so_far(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        def _slow_obj(fim, objective_option):
            time.sleep(0.003)
            return float(np.trace(fim))

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            with patch.object(
                DesignOfExperiments,
                "_evaluate_objective_for_option",
                side_effect=_slow_obj,
            ):
                with self.assertLogs(
                    "pyomo.contrib.doe.doe", level="WARNING"
                ) as log_cm:
                    points, diag = doe._lhs_initialize_experiments(
                        lhs_n_samples=6,
                        lhs_seed=12,
                        n_exp=2,
                        lhs_combo_parallel=True,
                        lhs_n_workers=2,
                        lhs_combo_chunk_size=5,
                        lhs_combo_parallel_threshold=1,
                        lhs_max_wall_clock_time=0.001,
                    )

        self.assertEqual(len(points), 2)
        self.assertTrue(any("time budget" in msg for msg in log_cm.output))
        self.assertTrue(diag["timed_out"])

    def test_lhs_combo_parallel_timeout_cancels_pending(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        def _slow_obj(fim, objective_option):
            time.sleep(0.01)
            return float(np.trace(fim))

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            with patch.object(
                DesignOfExperiments,
                "_evaluate_objective_for_option",
                side_effect=_slow_obj,
            ):
                with self.assertLogs(
                    "pyomo.contrib.doe.doe", level="WARNING"
                ) as log_cm:
                    points, diag = doe._lhs_initialize_experiments(
                        lhs_n_samples=9,
                        lhs_seed=4,
                        n_exp=2,
                        lhs_combo_parallel=True,
                        lhs_n_workers=2,
                        lhs_combo_chunk_size=1,
                        lhs_combo_parallel_threshold=1,
                        lhs_max_wall_clock_time=0.02,
                    )

        self.assertEqual(len(points), 2)
        self.assertTrue(diag["timed_out"])
        self.assertTrue(any("time budget" in msg for msg in log_cm.output))

    def test_lhs_combo_parallel_submit_loop_timeout_sets_timed_out(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=1)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        times = iter([0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            with patch(
                "pyomo.contrib.doe.doe.time.perf_counter",
                side_effect=lambda: next(times),
            ):
                points, diag = doe._lhs_initialize_experiments(
                    lhs_n_samples=1,
                    lhs_seed=19,
                    n_exp=1,
                    lhs_combo_parallel=True,
                    lhs_n_workers=2,
                    lhs_combo_chunk_size=1,
                    lhs_combo_parallel_threshold=1,
                    lhs_max_wall_clock_time=0.001,
                )

        self.assertEqual(len(points), 1)
        self.assertTrue(diag["timed_out"])

    def test_lhs_combo_parallel_deadline_cancels_pending_futures(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        stage = {"timeout": False}

        class _FakeFuture:
            def __init__(self, payload):
                self._payload = payload
                self.cancelled = False

            def result(self):
                return self._payload

            def cancel(self):
                self.cancelled = True
                return True

        class _FakeExecutor:
            def __init__(self, max_workers=None):
                self.created = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                # Return deterministic chunk scores; first future will be "done".
                idx = len(self.created)
                payload = (
                    (10.0 - idx, (0, 1), False)
                    if idx == 0
                    else (9.0 - idx, (0, 2), False)
                )
                fut = _FakeFuture(payload)
                self.created.append(fut)
                return fut

        def _fake_wait(pending, return_when=None):
            pending_list = list(pending)
            done = {pending_list[0]}
            still_pending = set(pending_list[1:])
            stage["timeout"] = True
            return done, still_pending

        def _fake_perf_counter():
            return 1.0 if stage["timeout"] else 0.0

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            with patch("pyomo.contrib.doe.doe._cf.ThreadPoolExecutor", _FakeExecutor):
                with patch("pyomo.contrib.doe.doe._cf.wait", side_effect=_fake_wait):
                    with patch(
                        "pyomo.contrib.doe.doe.time.perf_counter",
                        side_effect=_fake_perf_counter,
                    ):
                        points, diag = doe._lhs_initialize_experiments(
                            lhs_n_samples=3,
                            lhs_seed=23,
                            n_exp=2,
                            lhs_combo_parallel=True,
                            lhs_n_workers=2,
                            lhs_combo_chunk_size=1,
                            lhs_combo_parallel_threshold=1,
                            lhs_max_wall_clock_time=0.5,
                        )

        self.assertEqual(len(points), 2)
        self.assertTrue(diag["timed_out"])

    def test_lhs_combo_parallel_minimize_objective_update(self):
        doe = self._make_template_doe("trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, 3.0 * x + 1.0]])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_serial, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5, lhs_seed=52, n_exp=2, lhs_combo_parallel=False
            )

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_parallel, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5,
                lhs_seed=52,
                n_exp=2,
                lhs_combo_parallel=True,
                lhs_n_workers=2,
                lhs_combo_chunk_size=2,
                lhs_combo_parallel_threshold=1,
            )

        serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
        parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
        self.assertEqual(serial_norm, parallel_norm)

    def test_lhs_combo_serial_minimize_objective_update(self):
        doe = self._make_template_doe("trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)
        lhs_n_samples = 4
        lhs_seed = 61

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 0.25, 0.0], [0.0, 2.5 * x + 0.25]])

        first_exp_block = doe.model.param_scenario_blocks[0].exp_blocks[0]
        exp_input_vars = doe._get_experiment_input_vars(first_exp_block)
        lb_vals = np.array([v.lb for v in exp_input_vars])
        ub_vals = np.array([v.ub for v in exp_input_vars])

        rng = np.random.default_rng(lhs_seed)
        from scipy.stats.qmc import LatinHypercube

        per_dim_samples = []
        for i in range(len(exp_input_vars)):
            dim_seed = int(rng.integers(0, 2**31))
            sampler = LatinHypercube(d=1, seed=dim_seed)
            s_unit = sampler.random(n=lhs_n_samples).flatten()
            s_scaled = lb_vals[i] + s_unit * (ub_vals[i] - lb_vals[i])
            per_dim_samples.append(s_scaled.tolist())
        candidate_points = list(product(*per_dim_samples))
        candidate_fims = [_fake_fim(0, pt) for pt in candidate_points]

        best_combo = None
        best_obj = np.inf
        for combo in combinations(range(len(candidate_points)), 2):
            fim_total = sum((candidate_fims[idx] for idx in combo), np.zeros((2, 2)))
            obj_val = doe._evaluate_objective_from_fim(fim_total)
            if obj_val < best_obj:
                best_obj = obj_val
                best_combo = combo
        expected_points = [list(candidate_points[i]) for i in best_combo]

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            got_points, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=lhs_n_samples,
                lhs_seed=lhs_seed,
                n_exp=2,
                lhs_combo_parallel=False,
            )

        got_norm = sorted(tuple(np.round(p, 8)) for p in got_points)
        exp_norm = sorted(tuple(np.round(p, 8)) for p in expected_points)
        self.assertEqual(got_norm, exp_norm)

    def test_lhs_score_chunk_minimize_branch(self):
        doe = self._make_template_doe("trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=3)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 0.75, 0.0], [0.0, 2.0 * x + 0.75]])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_serial, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5, lhs_seed=77, n_exp=3, lhs_combo_parallel=False
            )

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_parallel, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5,
                lhs_seed=77,
                n_exp=3,
                lhs_combo_parallel=True,
                lhs_n_workers=2,
                lhs_combo_chunk_size=2,
                lhs_combo_parallel_threshold=1,
            )

        serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
        parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
        self.assertEqual(serial_norm, parallel_norm)

    def test_lhs_combo_no_scored_combo_falls_back_to_first_n_exp(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=3)
        lhs_n_samples = 3
        unit_samples = np.array([0.2, 0.5, 0.8])

        class _FakeLHS:
            def __init__(self, d, seed=None):
                self.d = d

            def random(self, n):
                assert self.d == 1
                assert n == lhs_n_samples
                return unit_samples.reshape((n, 1))

        first_exp_block = doe.model.param_scenario_blocks[0].exp_blocks[0]
        exp_input_vars = doe._get_experiment_input_vars(first_exp_block)
        lb = float(exp_input_vars[0].lb)
        ub = float(exp_input_vars[0].ub)
        expected_points = [[float(lb + s * (ub - lb))] for s in unit_samples]

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 1.0]])

        with patch("pyomo.contrib.doe.doe.LatinHypercube", _FakeLHS):
            with patch("pyomo.contrib.doe.doe._combinations", return_value=iter(())):
                with patch.object(
                    doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim
                ):
                    with self.assertLogs(
                        "pyomo.contrib.doe.doe", level="WARNING"
                    ) as log_cm:
                        got_points, _ = doe._lhs_initialize_experiments(
                            lhs_n_samples=lhs_n_samples,
                            lhs_seed=13,
                            n_exp=3,
                            lhs_combo_parallel=False,
                        )

        got_norm = sorted(tuple(np.round(p, 8)) for p in got_points)
        exp_norm = sorted(tuple(np.round(p, 8)) for p in expected_points)
        self.assertEqual(got_norm, exp_norm)
        self.assertTrue(
            any(
                "Falling back to the first n_exp candidate points." in msg
                for msg in log_cm.output
            )
        )

    def test_lhs_fim_evaluation_timeout_stops_early(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)

        def _slow_fim(experiment_index, input_values):
            time.sleep(0.01)
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, x + 2.0]])

        with patch.object(
            doe, "_compute_fim_at_point_no_prior", side_effect=_slow_fim
        ) as p:
            points, diag = doe._lhs_initialize_experiments(
                lhs_n_samples=10,
                lhs_seed=3,
                n_exp=2,
                lhs_parallel=False,
                lhs_combo_parallel=False,
                lhs_max_wall_clock_time=0.03,
            )

        self.assertEqual(len(points), 2)
        self.assertTrue(diag["timed_out"])
        self.assertLess(p.call_count, 10)
        self.assertLess(diag["n_candidates"], 10)

    def test_lhs_combo_scoring_n_exp_3_parallel_matches_serial(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=3)

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, 0.5 * x + 2.0]])

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_serial, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5, lhs_seed=1234, n_exp=3, lhs_combo_parallel=False
            )

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            points_parallel, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=5,
                lhs_seed=1234,
                n_exp=3,
                lhs_combo_parallel=True,
                lhs_n_workers=2,
                lhs_combo_chunk_size=3,
                lhs_combo_parallel_threshold=1,
            )

        serial_norm = sorted(tuple(np.round(p, 8)) for p in points_serial)
        parallel_norm = sorted(tuple(np.round(p, 8)) for p in points_parallel)
        self.assertEqual(serial_norm, parallel_norm)
        self.assertEqual(len(points_serial), 3)

    def test_lhs_combo_scoring_n_exp_3_matches_oracle(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=3)
        lhs_n_samples = 5
        lhs_seed = 2

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, 2.0 * x + 0.5]])

        # Recreate the exact candidate points from LHS generation (independent
        # oracle for combination scoring logic).
        first_exp_block = doe.model.param_scenario_blocks[0].exp_blocks[0]
        exp_input_vars = doe._get_experiment_input_vars(first_exp_block)
        lb_vals = np.array([v.lb for v in exp_input_vars])
        ub_vals = np.array([v.ub for v in exp_input_vars])
        rng = np.random.default_rng(lhs_seed)
        from scipy.stats.qmc import LatinHypercube

        per_dim_samples = []
        for i in range(len(exp_input_vars)):
            dim_seed = int(rng.integers(0, 2**31))
            sampler = LatinHypercube(d=1, seed=dim_seed)
            s_unit = sampler.random(n=lhs_n_samples).flatten()
            s_scaled = lb_vals[i] + s_unit * (ub_vals[i] - lb_vals[i])
            per_dim_samples.append(s_scaled.tolist())
        candidate_points = list(product(*per_dim_samples))

        # Oracle over all combinations of size 3.
        fims = [_fake_fim(0, pt) for pt in candidate_points]
        best_obj = -np.inf
        best_combo = None
        for combo in combinations(range(len(candidate_points)), 3):
            fim_total = sum((fims[i] for i in combo), np.zeros((2, 2)))
            obj = float(np.trace(fim_total))
            if obj > best_obj:
                best_obj = obj
                best_combo = combo
        expected_points = [list(candidate_points[i]) for i in best_combo]

        with patch.object(doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim):
            got_points, _ = doe._lhs_initialize_experiments(
                lhs_n_samples=lhs_n_samples,
                lhs_seed=lhs_seed,
                n_exp=3,
                lhs_combo_parallel=False,
            )

        got_norm = sorted(tuple(np.round(p, 8)) for p in got_points)
        exp_norm = sorted(tuple(np.round(p, 8)) for p in expected_points)
        self.assertEqual(got_norm, exp_norm)

    def test_lhs_matches_independent_oracle_with_fixed_samples(self):
        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=2)
        lhs_n_samples = 3

        first_exp_block = doe.model.param_scenario_blocks[0].exp_blocks[0]
        exp_input_vars = doe._get_experiment_input_vars(first_exp_block)
        self.assertEqual(len(exp_input_vars), 1)
        lb = float(exp_input_vars[0].lb)
        ub = float(exp_input_vars[0].ub)

        unit_samples = np.array([0.1, 0.4, 0.9])
        scaled_points = [float(lb + s * (ub - lb)) for s in unit_samples]

        class _FakeLHS:
            def __init__(self, d, seed=None):
                self.d = d

            def random(self, n):
                assert self.d == 1
                assert n == lhs_n_samples
                return unit_samples.reshape((n, 1))

        def _fake_fim(experiment_index, input_values):
            x = float(input_values[0])
            return np.array([[x + 1.0, 0.0], [0.0, 2.0 * x + 1.0]])

        with patch("pyomo.contrib.doe.doe.LatinHypercube", _FakeLHS):
            with patch.object(
                doe, "_compute_fim_at_point_no_prior", side_effect=_fake_fim
            ):
                got_points, _ = doe._lhs_initialize_experiments(
                    lhs_n_samples=lhs_n_samples,
                    lhs_seed=123,
                    n_exp=2,
                    lhs_combo_parallel=False,
                )

        # Independent brute-force oracle over explicit candidate points
        best_obj = -np.inf
        best_combo = None
        for combo in combinations(range(len(scaled_points)), 2):
            f1 = _fake_fim(0, [scaled_points[combo[0]]])
            f2 = _fake_fim(0, [scaled_points[combo[1]]])
            obj_val = float(np.trace(f1 + f2))
            if obj_val > best_obj:
                best_obj = obj_val
                best_combo = combo
        expected_points = [[scaled_points[i]] for i in best_combo]

        got_norm = sorted(tuple(np.round(p, 8)) for p in got_points)
        exp_norm = sorted(tuple(np.round(p, 8)) for p in expected_points)
        self.assertEqual(got_norm, exp_norm)

    def test_optimize_experiments_determinant_expected_values(self):
        # Match the multi-experiment example style (explicit experiment list)
        exp_list = [
            RooneyBieglerMultiExperiment(hour=1.0, y=8.3),
            RooneyBieglerMultiExperiment(hour=2.0, y=10.3),
        ]
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000

        doe = DesignOfExperiments(
            experiment_list=exp_list,
            objective_option="determinant",
            step=1e-2,
            solver=solver,
        )
        doe.optimize_experiments()

        scenario = doe.results["Scenarios"][0]
        got_hours = sorted(
            exp["Experiment Design"][0] for exp in scenario["Experiments"]
        )
        expected_hours = [1.9321985035514362, 9.999999685577139]

        self.assertStructuredAlmostEqual(got_hours, expected_hours, abstol=1e-3)
        self.assertAlmostEqual(scenario["log10 D-opt"], 6.028152580313302, places=3)

    def test_optimize_experiments_trace_expected_values(self):
        # Match the multi-experiment example style (explicit experiment list)
        exp_list = [
            RooneyBieglerMultiExperiment(hour=1.0, y=8.3),
            RooneyBieglerMultiExperiment(hour=2.0, y=10.3),
        ]
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        # prior_FIM from data `hour = 1, y = 8.3` with default value of parameters, which
        # is theta = {'asymptote': 15, 'rate_constant': 0.5}
        prior_FIM = np.array(
            [[15.48181217, 357.97684273], [357.97684273, 8277.28811613]]
        )

        doe = DesignOfExperiments(
            experiment_list=exp_list,
            objective_option="trace",
            step=1e-2,
            solver=solver,
            prior_FIM=prior_FIM,
        )
        doe.optimize_experiments()

        scenario = doe.results["Scenarios"][0]
        got_hours = sorted(
            exp["Experiment Design"][0] for exp in scenario["Experiments"]
        )
        expected_hours = [10.0, 10.0]

        self.assertStructuredAlmostEqual(got_hours, expected_hours, abstol=1e-3)
        self.assertAlmostEqual(scenario["log10 A-opt"], -2.2347, places=3)

    def test_optimize_experiments_prior_fim_aggregation_non_lhs_template_mode(self):
        prior_fim = np.array([[2.0, 0.1], [0.1, 1.5]])
        doe = self._make_template_doe("pseudo_trace")
        doe.prior_FIM = prior_fim.copy()

        doe.optimize_experiments(n_exp=2, initialization_method=None)

        scenario = doe.results["Scenarios"][0]
        total_fim = np.array(scenario["Total FIM"])
        exp_fim_sum = sum(
            (np.array(exp_data["FIM"]) for exp_data in scenario["Experiments"]),
            np.zeros_like(total_fim),
        )
        stored_prior = np.array(doe.results["Prior FIM"])

        self.assertTrue(np.allclose(total_fim, exp_fim_sum + prior_fim, atol=1e-6))
        self.assertTrue(np.allclose(total_fim, exp_fim_sum + stored_prior, atol=1e-6))

    def test_optimize_experiments_prior_fim_aggregation_non_lhs_user_initialized_mode(
        self,
    ):
        exp_list = [
            RooneyBieglerMultiExperiment(hour=1.5, y=9.0),
            RooneyBieglerMultiExperiment(hour=3.5, y=12.0),
        ]
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        prior_fim = np.array([[1.25, 0.05], [0.05, 0.9]])

        doe = DesignOfExperiments(
            experiment_list=exp_list,
            objective_option="pseudo_trace",
            step=1e-2,
            solver=solver,
            prior_FIM=prior_fim,
        )
        doe.optimize_experiments(initialization_method=None)

        scenario = doe.results["Scenarios"][0]
        total_fim = np.array(scenario["Total FIM"])
        exp_fim_sum = sum(
            (np.array(exp_data["FIM"]) for exp_data in scenario["Experiments"]),
            np.zeros_like(total_fim),
        )
        stored_prior = np.array(doe.results["Prior FIM"])

        self.assertTrue(np.allclose(total_fim, exp_fim_sum + prior_fim, atol=1e-6))
        self.assertTrue(np.allclose(total_fim, exp_fim_sum + stored_prior, atol=1e-6))

    def test_optimize_experiments_safe_metric_failure_sets_nan(self):
        doe = self._make_template_doe("pseudo_trace")
        with patch(
            "pyomo.contrib.doe.doe.np.linalg.inv", side_effect=RuntimeError("boom")
        ):
            with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as log_cm:
                doe.optimize_experiments(n_exp=1)

        scenario = doe.results["Scenarios"][0]
        self.assertTrue(np.isnan(scenario["log10 A-opt"]))
        self.assertTrue(
            any("failed to compute log10 A-opt" in msg for msg in log_cm.output)
        )

    def test_optimize_experiments_non_cholesky_determinant_initialization(self):
        exp = RooneyBieglerMultiExperiment(hour=2.0, y=10.0)
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        doe = DesignOfExperiments(
            experiment_list=[exp],
            objective_option="determinant",
            step=1e-2,
            solver=solver,
            _Cholesky_option=False,
            _only_compute_fim_lower=False,
        )
        original_solve = doe.solver.solve

        class _MockSolverInfo:
            status = "ok"
            termination_condition = "optimal"
            message = "mock-solve"

        class _MockResults:
            solver = _MockSolverInfo()

        def _solve_real_for_square_then_mock(*args, **kwargs):
            model = args[0] if args else kwargs.get("model", None)
            if model is not None and hasattr(model, "dummy_obj"):
                # Keep square-solve path real so model state initializes correctly.
                return original_solve(*args, **kwargs)
            return _MockResults()

        with patch.object(
            doe.solver, "solve", side_effect=_solve_real_for_square_then_mock
        ):
            doe.optimize_experiments(n_exp=1)

        scenario_block = doe.model.param_scenario_blocks[0]
        self.assertTrue(hasattr(scenario_block.obj_cons, "determinant"))
        total_fim = np.array(doe.results["Scenarios"][0]["Total FIM"])
        expected_det = np.linalg.det(total_fim)
        self.assertAlmostEqual(
            pyo.value(scenario_block.obj_cons.determinant), expected_det, places=6
        )


if __name__ == "__main__":
    unittest.main()
