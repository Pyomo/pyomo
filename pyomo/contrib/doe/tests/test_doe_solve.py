# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
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

import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")


from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.examples.polynomial import (
    PolynomialExperiment,
    run_polynomial_doe,
)
from pyomo.contrib.doe.tests.experiment_class_example_flags import (
    RooneyBieglerExperimentBad,
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


def get_polynomial_experiment(measurement_error=1.0):
    """Build a fresh polynomial experiment with a configurable measurement error."""
    experiment = PolynomialExperiment()
    model = experiment.get_labeled_model()
    model.measurement_error[model.y] = measurement_error
    return experiment


def get_polynomial_args(
    gradient_method=None,
    measurement_error=1.0,
    prior_FIM=None,
    objective_option="determinant",
):
    """Return standard DoE arguments for the public polynomial example."""
    experiment = get_polynomial_experiment(measurement_error=measurement_error)
    DoE_args = get_standard_args(experiment, "central", objective_option)
    DoE_args["scale_nominal_param_value"] = False
    DoE_args["prior_FIM"] = prior_FIM
    if gradient_method is not None:
        DoE_args["gradient_method"] = gradient_method
    return DoE_args


def get_expected_polynomial_fim(x1=2.0, x2=3.0, measurement_error=1.0, prior_FIM=None):
    """Return the hand-derived polynomial Fisher information matrix."""
    sensitivity = np.array([x1, x2, x1 * x2, 1.0], dtype=float)
    fim = np.outer(sensitivity, sensitivity) / (measurement_error**2)
    if prior_FIM is not None:
        fim = fim + prior_FIM
    return fim


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

    # This legacy Cholesky/bad-prior case is kept disabled in
    # this PR. It is not part of the active regression signal, and this cleanup is
    # focused on replacing active general-purpose coverage with
    # Rooney-Biegler or polynomial examples rather than rewriting inactive,
    # branch-specific tests.
    def DISABLE_test_rooney_biegler_obj_cholesky_solve_bad_prior(self):
        # [10/2025] This test has been disabled because it frequently
        # (and randomly) returns "infeasible" when run on Windows.
        from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS

        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment()

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

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_compute_FIM_pynumero(self):
        fd_method = "central"
        obj_used = "zero"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["gradient_method"] = "pynumero"

        doe_obj = DesignOfExperiments(**DoE_args)

        expected_FIM = np.array(
            [[18957.7788694, 4238.27606876], [4238.27606876, 947.52577076]]
        )

        self.assertTrue(np.all(np.isclose(doe_obj.compute_FIM(), expected_FIM)))

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
    def test_polynomial_grid_search(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = PolynomialExperiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        # Use a small 2x2 polynomial design grid for a lightweight factorial check.
        design_ranges = {"x1": [0, 5, 2], "x2": [0, 5, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        # Check that the factorial results contain the expected 2x2 grid entries.
        x1_vals = doe_obj.fim_factorial_results["x1"]
        x2_vals = doe_obj.fim_factorial_results["x2"]

        # assert length is correct (2x2 = 4 evaluations)
        self.assertTrue((len(x1_vals) == 4) and (len(x2_vals) == 4))
        self.assertTrue((len(set(x1_vals)) == 2) and (len(set(x2_vals)) == 2))

        # Check that each polynomial design variable spans the requested grid values.
        self.assertTrue(
            (set(x1_vals).issuperset(set([0, 5])))
            and (set(x2_vals).issuperset(set([0, 5])))
        )

    def test_rooney_biegler_run_doe_pynumero(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["gradient_method"] = "pynumero"

        doe_obj = DesignOfExperiments(**DoE_args)
        # Rooney-Biegler determinant solves are numerically more stable with a
        # prior. Use the FIM at the current nominal design as the prior so this
        # symbolic run_doe() test exercises the intended backend without
        # relying on a singular starting information matrix.
        prior_FIM = doe_obj.compute_FIM()
        doe_obj.prior_FIM = prior_FIM
        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)
        self.assertTrue(np.all(np.isclose(FIM, L @ L.T)))
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q + prior_FIM)))

    def test_rooney_biegler_run_doe_determinant_regression(self):
        """Check a stable Rooney-Biegler optimum fingerprint against expected values."""
        experiment = get_rooney_biegler_experiment()
        DoE_args = get_standard_args(experiment, "central", "determinant")
        DoE_args["gradient_method"] = "pynumero"
        doe_obj = DesignOfExperiments(**DoE_args)
        # Rooney-Biegler determinant solves are numerically more stable with a
        # prior. Use the FIM at the current nominal design as the prior so this
        # symbolic regression test keeps the determinant problem well
        # conditioned while preserving the same backend configuration.
        prior_FIM = doe_obj.compute_FIM()
        doe_obj.prior_FIM = prior_FIM
        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")
        self.assertEqual(
            str(doe_obj.results["Termination Condition"]).lower(), "optimal"
        )

        design = doe_obj.results["Experiment Design"]
        self.assertAlmostEqual(design[0], 9.999999776254937, places=4)
        fim = np.array(doe_obj.results["FIM"])
        self.assertAlmostEqual(fim[0, 0], 41155.59271917, places=2)
        self.assertAlmostEqual(fim[1, 1], 973.06126181, places=2)
        self.assertAlmostEqual(
            doe_obj.results["log10 D-opt"],7.179982499524086, places=4
        )
        self.assertAlmostEqual(
            doe_obj.results["log10 A-opt"], -2.5554049159721415, places=4
        )

    

    def test_rooney_biegler_run_doe_pynumero_objective_matrix(self):
        """Exercise the symbolic Rooney-Biegler run_doe path across objective options."""
        test_cases = ["trace", "determinant", "zero"]

        for objective_option in test_cases:
            with self.subTest(objective_option=objective_option):
                experiment = get_rooney_biegler_experiment()
                DoE_args = get_standard_args(experiment, "central", objective_option)
                DoE_args["gradient_method"] = "pynumero"

                doe_obj = DesignOfExperiments(**DoE_args)
                # Rooney-Biegler run_doe() cases are more stable with the
                # nominal-design FIM supplied as a prior, so each objective is
                # exercised under the symbolic backend without starting from a
                # nearly singular information matrix.
                prior_FIM = doe_obj.compute_FIM()
                doe_obj.prior_FIM = prior_FIM
                doe_obj.run_doe()

                self.assertEqual(doe_obj.results["Solver Status"], "ok")

                FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)
                if objective_option == "determinant":
                    self.assertTrue(np.all(np.isclose(FIM, L @ L.T)))
                self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q + prior_FIM)))

    def test_polynomial_example_compute_fim_pynumero(self):
        """Check that the transplanted polynomial example computes the expected FIM."""
        fim = run_polynomial_doe()
        expected = get_expected_polynomial_fim()
        self.assertEqual(fim.shape, expected.shape)
        self.assertTrue(np.allclose(fim, expected))


    def test_polynomial_example_measurement_error_scaling(self):
        """Check that doubling the measurement error scales the FIM by one quarter."""
        fim_sigma_one = DesignOfExperiments(
            **get_polynomial_args(gradient_method="pynumero", measurement_error=1.0)
        ).compute_FIM()
        fim_sigma_two = DesignOfExperiments(
            **get_polynomial_args(gradient_method="pynumero", measurement_error=2.0)
        ).compute_FIM()

        self.assertTrue(np.allclose(fim_sigma_two, fim_sigma_one / 4.0))
        self.assertTrue(
            np.allclose(
                fim_sigma_two, get_expected_polynomial_fim(measurement_error=2.0)
            )
        )

    def test_polynomial_example_prior_fim_adds_directly(self):
        """Check that the polynomial example adds prior information entry-wise."""
        prior_FIM = np.diag([1.0, 2.0, 3.0, 4.0])
        doe_obj = DesignOfExperiments(
            **get_polynomial_args(
                gradient_method="pynumero", measurement_error=1.0, prior_FIM=prior_FIM
            )
        )
        expected = get_expected_polynomial_fim(prior_FIM=prior_FIM)

        self.assertTrue(np.allclose(doe_obj.compute_FIM(), expected))

    def test_polynomial_example_run_doe_smoke(self):
        """Check that the public polynomial example can solve a tiny DoE problem.
        Also do a regression test to check that the solution returned stays correct over time"""
        prior_FIM = np.eye(4)
        doe_obj = DesignOfExperiments(
            **get_polynomial_args(
                gradient_method="pynumero",
                prior_FIM=prior_FIM,
                objective_option="determinant",
            )
        )

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")
        self.assertEqual(str(doe_obj.results["Termination Condition"]).lower(), "optimal")
        design = doe_obj.results["Experiment Design"]
        self.assertAlmostEqual(design[0],5.0,places=4)
        self.assertAlmostEqual(design[1],5.0,places =4)

        self.assertAlmostEqual(doe_obj.results["log10 D-opt"],2.830588683545922, places=4)

        fim = np.array(doe_obj.results["FIM"])
        self.assertAlmostEqual(fim[0,0], 26.00000045, places=4)
        self.assertAlmostEqual(fim[3,3], 2.0, places=4)

    def test_rescale_FIM(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment()

        # With parameter scaling
        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        # Without parameter scaling
        DoE_args2 = get_standard_args(experiment, fd_method, obj_used)
        DoE_args2["scale_nominal_param_value"] = False

        doe_obj2 = DesignOfExperiments(**DoE_args2)
        # Compare scaled and unscaled FIMs at the same nominal design. For the
        # Rooney-Biegler replacement, this avoids introducing determinant-solve
        # instability that is unrelated to the rescaling utility itself.
        FIM = doe_obj.compute_FIM()
        FIM2 = doe_obj2.compute_FIM()

        # Get rescaled FIM from the scaled version
        param_vals = np.array(
            [
                [
                    v
                    for k, v in experiment.get_labeled_model().unknown_parameters.items()
                ]
            ]
        )

        resc_FIM = rescale_FIM(FIM, param_vals)

        # Compare scaled and rescaled values
        self.assertTrue(np.all(np.isclose(FIM2, resc_FIM)))

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_solve_bad_model(self):
        fd_method = "central"
        obj_used = "determinant"

        # Use the Rooney-Biegler bad example as the lightweight bad-model case.
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
    def test_rooney_biegler_grid_search_bad_model(self):
        fd_method = "central"
        obj_used = "determinant"

        # Use the Rooney-Biegler bad example as the lightweight bad-model case.
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
    def test_polynomial_full_factorial(self):
        """Check 2D factorial FIM metrics on the lightweight polynomial example."""
        log10_D_opt_expected = [
            3.771625936657566,
            5.566143287412265,
            5.910363131426261,
            6.173537519214883,
        ]

        log10_A_opt_expected = [
            3.771846315265457,
            5.5661468254334245,
            5.910364732979566,
            6.173538392929174,
        ]

        log10_E_opt_expected = [
            -4.821637332766436e-17,
            -7.8206957537542e-13,
            -2.4960652144337014e-12,
            -1.0111706376862954e-10,
        ]

        log10_ME_opt_expected = [
            3.771625936657538,
            5.566143287414164,
            5.910363131438886,
            6.173537519245602,
        ]

        eigval_min_expected = [
            0.9999999999999999,
            0.9999999999981992,
            0.9999999999942526,
            0.9999999997671694,
        ]

        eigval_max_expected = [
            5910.523340060432,
            368250.4510100104,
            813510.4413073618,
            1491205.5769174611,
        ]

        det_FIM_expected = [
            5910.523340060814,
            368250.45101058815,
            813510.4413089894,
            1491205.5769048228,
        ]

        trace_FIM_expected = [
            5913.523340060433,
            368253.4510100104,
            813513.4413073619,
            1491208.5769174611,
        ]
        experiment = PolynomialExperiment()
        DoE_args = get_standard_args(experiment, "central", "trace")
        DoE_args["scale_nominal_param_value"] = False
        # The polynomial model has one output and four parameters, so the raw
        # outer-product FIM is rank one. Seed the factorial sweep with an
        # identity prior to keep the metric regressions positive definite.
        DoE_args["prior_FIM"] = np.eye(4)

        ff = DesignOfExperiments(**DoE_args)
        ff.compute_FIM_full_factorial(design_ranges={"x1": [0, 5, 2], "x2": [0, 5, 2]})

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

    @unittest.skipUnless(pandas_available, "test requires pandas")
    @unittest.skipUnless(ipopt_available, "test requires ipopt")
    def test_rooney_biegler_factorial_results_dataframe_schema(self):
        """Check the schema of the Rooney-Biegler factorial-results table."""
        experiment = run_rooney_biegler_doe()["experiment"]
        DoE_args = get_standard_args(experiment, "central", "trace")
        DoE_args["gradient_method"] = "pynumero"
        doe_obj = DesignOfExperiments(**DoE_args)

        results = doe_obj.compute_FIM_full_factorial(design_ranges={"hour": [0, 10, 3]})
        results_pd = pd.DataFrame(results)

        expected_columns = {
            "hour",
            "log10 D-opt",
            "log10 A-opt",
            "log10 pseudo A-opt",
            "log10 E-opt",
            "log10 ME-opt",
            "eigval_min",
            "eigval_max",
            "det_FIM",
            "trace_cov",
            "trace_FIM",
            "solve_time",
        }

        self.assertTrue(expected_columns.issubset(results_pd.columns))
        self.assertEqual(len(results_pd), 3)
        self.assertEqual(sorted(results_pd["hour"].tolist()), [0.0, 5.0, 10.0])
        self.assertTrue(np.all(np.isfinite(results_pd["solve_time"])))

    @unittest.skipUnless(pandas_available, "test requires pandas")
    @unittest.skipUnless(ipopt_available, "test requires ipopt")
    def test_draw_factorial_figure_accepts_dataframe_input(self):
        """Check draw_factorial_figure accepts a DataFrame and stores filtered rows."""
        doe_obj = DesignOfExperiments(
            **get_polynomial_args(gradient_method = "pynumero", objective_option="determinant")
        )

        results = doe_obj.compute_FIM_full_factorial(
            design_ranges={"x1": [0, 5, 2], "x2": [0, 5, 2]}
        )
        results_pd = pd.DataFrame(results)

        doe_obj.draw_factorial_figure(
            results=results_pd,
            sensitivity_design_variables=["x1"],
            fixed_design_variables={"x2": 0.0},
            full_design_variable_names=["x1", "x2"],
            log_scale=False,
            figure_file_name=None,
        )

        filtered = doe_obj.figure_result_data
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertEqual(len(filtered), 2)
        self.assertTrue(np.allclose(filtered["x2"].values, 0.0))
        self.assertEqual(sorted(filtered["x1"].tolist()), [0.0, 5.0])

    @unittest.skipUnless(pandas_available, "test requires pandas")
    @unittest.skipUnless(ipopt_available, "test requires ipopt")
    def test_draw_factorial_figure_bad_fixed_variable_raises(self):
        """Check draw_factorial_figure rejects unknown fixed design variables."""
        doe_obj = DesignOfExperiments(
            **get_polynomial_args(gradient_method = "pynumero", objective_option="determinant")
        )

        results = doe_obj.compute_FIM_full_factorial(design_ranges={"x1": [0,5,3], "x2":[0,5,3]})

        with self.assertRaisesRegex(
            ValueError, "Fixed design variables do not all appear"
        ):
            doe_obj.draw_factorial_figure(
                results=results,
                sensitivity_design_variables=["x1"],
                fixed_design_variables={"bad_name": 5.0},
                full_design_variable_names=["x1","x2"],
                log_scale=False,
                figure_file_name=None,
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

    def test_polynomial_2D_plotting_function(self):
        # Use the lightweight polynomial example for generic 2D factorial plotting.
        plt = matplotlib.pyplot

        # File prefix for saved plots
        prefix_linear = "polynomial_linear"
        prefix_log = "polynomial_log"

        # Clean up any existing plot files from test runs
        def cleanup_files():
            files_to_remove = glob("polynomial_*.png")
            for f in files_to_remove:
                try:
                    os.remove(f)
                except OSError:
                    pass
            plt.close('all')

        self.addCleanup(cleanup_files)

        experiment = PolynomialExperiment()
        DoE_args = get_standard_args(experiment, "central","determinant")
        DoE_args["gradient_method"] = "pynumero"
        DoE_args["scale_nominal_param_value"] = False

        doe_obj = DesignOfExperiments(**DoE_args)
        # Build polynomial factorial results and draw the linear-scale 2D plots.
        doe_obj.compute_FIM_full_factorial(design_ranges={"x1": [0, 5, 2], "x2": [0, 5, 2]})
        doe_obj.draw_factorial_figure(
            sensitivity_design_variables=["x1", "x2"],
            fixed_design_variables={},
            full_design_variable_names=["x1", "x2"],
            figure_file_name=prefix_linear,
            log_scale=False,
        )

        # Verify that the linear scale plots were also created
        # Check that we found exactly 5 files (A, D, E, ME, pseudo_A)
        expected_plot_linear = glob(f"{prefix_linear}*.png")
        self.assertTrue(
            len(expected_plot_linear) == 5,
            f"Expected 5 plot files, but found {len(expected_plot_linear)}. Files found: {expected_plot_linear}",
        )

        # Reuse the same factorial results to draw the log-scale 2D plots.
        doe_obj.draw_factorial_figure(
            sensitivity_design_variables=["x1", "x2"],
            fixed_design_variables={},
            full_design_variable_names=["x1", "x2"],
            figure_file_name=prefix_log,
            log_scale=True,
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
