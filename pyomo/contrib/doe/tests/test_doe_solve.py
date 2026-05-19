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
from glob import glob

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
    )
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
        RooneyBieglerExperiment,
    )
from pyomo.contrib.doe.utils import rescale_FIM, compute_FIM_metrics
from pyomo.contrib.doe.examples.rooney_biegler_doe_example import run_rooney_biegler_doe

import pyomo.environ as pyo

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()
k_aug_available = SolverFactory("k_aug", solver_io="nl", validate=False)

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
        1 / v**2 for k, v in model.scenario_blocks[0].measurement_error.items()
    ]
    param_vals = np.array(
        [[v for k, v in model.scenario_blocks[0].unknown_parameters.items()]]
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
    args['experiment'] = experiment
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
        not k_aug_available.available(False), "The 'k_aug' command is not available"
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
                    for k, v in doe_obj.model.scenario_blocks[
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
            n_points_for_C0=2,
            n_points_for_T0=2,
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


class TestComputeFIMFactorial(unittest.TestCase):
    def _assert_factorial_fim_matches_pointwise_fim(
        self, doe_obj, experiment, design_var, factorial_results
    ):
        """Validate that factorial FIMs match direct pointwise compute_FIM results."""
        for idx, design_point in enumerate(factorial_results[design_var.name]):
            point_model = experiment.get_labeled_model().clone()
            point_var = pyo.ComponentUID(design_var).find_component_on(point_model)
            point_var.fix(design_point)

            point_fim = doe_obj.compute_FIM(model=point_model, method="sequential")
            factorial_fim = np.array(factorial_results["FIM_all"][idx])
            self.assertTrue(np.allclose(factorial_fim, point_fim, rtol=1e-7, atol=1e-7))

    def test_compute_fim_factorial_matches_pointwise_compute_fim(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()
        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["logger_level"] = logging.ERROR
        doe_obj = DesignOfExperiments(**DoE_args)

        base_model = experiment.get_labeled_model()
        design_var = next(iter(base_model.experiment_inputs.keys()))
        design_points = [1.0, 5.0, 9.0]
        design_vals = pyo.ComponentMap(((design_var, design_points),))

        factorial_results = doe_obj.compute_FIM_factorial(
            design_vals=design_vals,
            method="sequential",
            return_df=False,
            traversal_scheme="nested_for_loop",
        )

        self.assertEqual(factorial_results["total_points"], len(design_points))
        self.assertEqual(factorial_results["success_count"], len(design_points))
        self.assertEqual(factorial_results["failure_count"], 0)

        for idx, design_point in enumerate(design_points):
            point_model = experiment.get_labeled_model().clone()
            point_var = pyo.ComponentUID(design_var).find_component_on(point_model)
            point_var.fix(design_point)

            point_fim = doe_obj.compute_FIM(model=point_model, method="sequential")
            factorial_fim = np.array(factorial_results["FIM_all"][idx])

            self.assertTrue(np.allclose(factorial_fim, point_fim, rtol=1e-7, atol=1e-7))

            (
                det_FIM,
                trace_cov,
                trace_FIM,
                E_vals,
                _,
                D_opt,
                A_opt,
                pseudo_A_opt,
                E_opt,
                ME_opt,
            ) = compute_FIM_metrics(point_fim)

            self.assertTrue(
                np.isclose(
                    factorial_results["log10 D-opt"][idx],
                    D_opt,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["log10 A-opt"][idx],
                    A_opt,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["log10 pseudo A-opt"][idx],
                    pseudo_A_opt,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["log10 E-opt"][idx],
                    E_opt,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["log10 ME-opt"][idx],
                    ME_opt,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["eigval_min"][idx],
                    np.min(E_vals),
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["eigval_max"][idx],
                    np.max(E_vals),
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["det_FIM"][idx],
                    det_FIM,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["trace_cov"][idx],
                    trace_cov,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                np.isclose(
                    factorial_results["trace_FIM"][idx],
                    trace_FIM,
                    rtol=1e-7,
                    atol=1e-7,
                    equal_nan=True,
                )
            )

    def test_compute_fim_factorial_step_formula_uses_design_span(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()
        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["logger_level"] = logging.ERROR
        doe_obj = DesignOfExperiments(**DoE_args)

        factorial_results = doe_obj.compute_FIM_factorial(
            abs_step=[2.0],
            rel_step=[0.1],
            method="sequential",
            return_df=False,
            traversal_scheme="nested_for_loop",
        )

        # Rooney-Biegler hour bounds are [0, 10], so:
        # delta = (ub - lb) * rel_change + abs_change = 10*0.1 + 2 = 3
        expected_hour_points = [0.0, 3.0, 6.0, 9.0]
        self.assertStructuredAlmostEqual(
            factorial_results["hour"], expected_hour_points, abstol=1e-8, reltol=1e-8
        )
        self.assertEqual(factorial_results["total_points"], len(expected_hour_points))
        self._assert_factorial_fim_matches_pointwise_fim(
            doe_obj=doe_obj,
            experiment=experiment,
            design_var=next(
                iter(experiment.get_labeled_model().experiment_inputs.keys())
            ),
            factorial_results=factorial_results,
        )

    def test_compute_fim_factorial_supported_modes_generate_expected_points(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()
        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["logger_level"] = logging.ERROR
        doe_obj = DesignOfExperiments(**DoE_args)

        base_model = experiment.get_labeled_model()
        design_var = next(iter(base_model.experiment_inputs.keys()))
        design_var_cuid = pyo.ComponentUID(design_var)

        # Each case exercises a distinct public input mode and checks the actual
        # sampled design points from compute_FIM_factorial.
        mode_cases = [
            (
                "design_vals_component_uid",
                {"design_vals": {design_var_cuid: [2.0, 6.0, 10.0]}},
                [2.0, 6.0, 10.0],
            ),
            ("n_design_points", {"n_design_points": 5}, [0.0, 2.5, 5.0, 7.5, 10.0]),
            ("abs_step_only", {"abs_step": [4.0]}, [0.0, 4.0, 8.0]),
            ("rel_step_only", {"rel_step": [0.25]}, [0.0, 2.5, 5.0, 7.5, 10.0]),
            (
                "abs_step_rel_step",
                {"abs_step": [2.0], "rel_step": [0.1]},
                [0.0, 3.0, 6.0, 9.0],
            ),
        ]

        for _, mode_kwargs, expected_points in mode_cases:
            factorial_results = doe_obj.compute_FIM_factorial(
                method="sequential",
                return_df=False,
                traversal_scheme="nested_for_loop",
                **mode_kwargs,
            )

            self.assertStructuredAlmostEqual(
                factorial_results["hour"], expected_points, abstol=1e-6, reltol=1e-6
            )
            self.assertEqual(factorial_results["total_points"], len(expected_points))
            self.assertEqual(factorial_results["success_count"], len(expected_points))
            self.assertEqual(factorial_results["failure_count"], 0)
            self._assert_factorial_fim_matches_pointwise_fim(
                doe_obj=doe_obj,
                experiment=experiment,
                design_var=design_var,
                factorial_results=factorial_results,
            )

    def test_compute_fim_factorial_nested_for_loop_order_on_two_by_two_grid(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)
        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["logger_level"] = logging.ERROR
        doe_obj = DesignOfExperiments(**DoE_args)

        model = experiment.get_labeled_model()
        design_components = {comp.name: comp for comp in model.experiment_inputs.keys()}
        ca0 = design_components["CA[0]"]
        t0 = design_components["T[0]"]

        # Explicit 2x2 grid using component keys so we can verify nested-loop order.
        design_vals = pyo.ComponentMap(((ca0, [1.0, 5.0]), (t0, [300.0, 700.0])))

        factorial_results = doe_obj.compute_FIM_factorial(
            design_vals=design_vals,
            method="sequential",
            return_df=False,
            traversal_scheme="nested_for_loop",
        )

        # Hardcoded expected order for nested-for-loop traversal:
        # first CA[0], then T[0] (product order).
        self.assertStructuredAlmostEqual(
            factorial_results["CA[0]"], [1.0, 1.0, 5.0, 5.0], abstol=1e-8, reltol=1e-8
        )
        self.assertStructuredAlmostEqual(
            factorial_results["T[0]"],
            [300.0, 700.0, 300.0, 700.0],
            abstol=1e-8,
            reltol=1e-8,
        )
        self.assertEqual(factorial_results["total_points"], 4)
        self.assertEqual(factorial_results["success_count"], 4)
        self.assertEqual(factorial_results["failure_count"], 0)


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
            n_points_for_C0=1,
            n_points_for_T0=1,
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
            n_points_for_C0=1,
            n_points_for_T0=1,
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


if __name__ == "__main__":
    unittest.main()
