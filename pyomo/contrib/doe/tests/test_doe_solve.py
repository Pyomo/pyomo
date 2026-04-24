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
import subprocess
import tempfile
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

import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.examples.polynomial import (
    PolynomialExperiment,
    run_polynomial_doe,
)
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


def _optimize_experiments_param_scenario(results, index=0):
    """Return one parameter-scenario entry from optimize_experiments() results."""
    return results["solution"]["param_scenarios"][index]


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
    args['experiment'] = None if experiment is None else [experiment]
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

    def test_compute_FIM_multi_experiment_is_sum_of_experiment_fims(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        fim_exp1 = np.array([[12.0, 3.0], [3.0, 8.0]])
        fim_exp2 = np.array([[5.0, 2.0], [2.0, 4.0]])
        prior_fim = np.array([[1.5, 0.1], [0.1, 0.5]])

        multi_args = get_standard_args(
            RooneyBieglerMultiExperiment(hour=1.5, y=9.0), fd_method, obj_used
        )
        multi_args["experiment"] = [
            RooneyBieglerMultiExperiment(hour=1.5, y=9.0),
            RooneyBieglerMultiExperiment(hour=3.5, y=12.0),
        ]
        multi_args["prior_FIM"] = prior_fim
        doe_multi = DesignOfExperiments(**multi_args)

        def _fake_sequential_fim(*args, **kwargs):
            # Return deterministic per-experiment FIMs keyed by the fixed
            # experiment input so the aggregation can be asserted exactly.
            model = kwargs.get("model")
            hour = float(pyo.value(model.hour))
            if np.isclose(hour, 1.5):
                doe_multi.seq_FIM = fim_exp1.copy()
            elif np.isclose(hour, 3.5):
                doe_multi.seq_FIM = fim_exp2.copy()
            else:
                raise RuntimeError(f"Unexpected hour value in mocked test: {hour}")

        with patch.object(
            doe_multi, "_sequential_FIM", side_effect=_fake_sequential_fim
        ):
            fim_total = doe_multi.compute_FIM(method="sequential")

        fim_expected = fim_exp1 + fim_exp2 + prior_fim
        self.assertTrue(np.allclose(fim_total, fim_expected, atol=1e-12))
        self.assertEqual(len(doe_multi._computed_FIM_by_experiment), 2)
        self.assertTrue(
            np.allclose(doe_multi._computed_FIM_by_experiment[0], fim_exp1, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(doe_multi._computed_FIM_by_experiment[1], fim_exp2, atol=1e-12)
        )

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
            doe_obj.results["log10 D-opt"], 7.179982499524086, places=4
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
                self.assertTrue(
                    np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q + prior_FIM))
                )

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
        Also do a regression test to check that the solution returned stays correct over time
        """
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
        self.assertEqual(
            str(doe_obj.results["Termination Condition"]).lower(), "optimal"
        )
        design = doe_obj.results["Experiment Design"]
        self.assertAlmostEqual(design[0], 5.0, places=4)
        self.assertAlmostEqual(design[1], 5.0, places=4)

        self.assertAlmostEqual(
            doe_obj.results["log10 D-opt"], 2.830588683545922, places=4
        )

        fim = np.array(doe_obj.results["FIM"])
        self.assertAlmostEqual(fim[0, 0], 26.00000045, places=4)
        self.assertAlmostEqual(fim[3, 3], 2.0, places=4)

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
                    for k, v in doe_obj.compute_FIM_model.unknown_parameters.items()
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
        with tempfile.TemporaryDirectory() as tmpdir:
            file_prefix = os.path.join(tmpdir, "rooney_biegler")
            prev_cwd = os.getcwd()
            os.chdir(tmpdir)
            self.addCleanup(os.chdir, prev_cwd)

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
            expected_log10_D_opt = [
                6.583798747893548,
                6.691228337572129,
                6.864794726228617,
            ]
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
                results_dict["log10 pseudo A-opt"],
                expected_log10_pseudo_A_opt,
                abstol=1e-4,
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
            **get_polynomial_args(
                gradient_method="pynumero", objective_option="determinant"
            )
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
            **get_polynomial_args(
                gradient_method="pynumero", objective_option="determinant"
            )
        )

        results = doe_obj.compute_FIM_full_factorial(
            design_ranges={"x1": [0, 5, 3], "x2": [0, 5, 3]}
        )

        with self.assertRaisesRegex(
            ValueError, "Fixed design variables do not all appear"
        ):
            doe_obj.draw_factorial_figure(
                results=results,
                sensitivity_design_variables=["x1"],
                fixed_design_variables={"bad_name": 5.0},
                full_design_variable_names=["x1", "x2"],
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

        with tempfile.TemporaryDirectory() as tmpdir:
            # File prefix for saved plots
            # Define prefixes for the two runs
            prefix_linear = os.path.join(tmpdir, "rooney_linear")
            prefix_log = os.path.join(tmpdir, "rooney_log")

            self.addCleanup(plt.close, 'all')

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

        with tempfile.TemporaryDirectory() as tmpdir:
            # File prefix for saved plots
            prefix_linear = os.path.join(tmpdir, "polynomial_linear")
            prefix_log = os.path.join(tmpdir, "polynomial_log")

            self.addCleanup(plt.close, 'all')

            experiment = PolynomialExperiment()
            DoE_args = get_standard_args(experiment, "central", "determinant")
            DoE_args["gradient_method"] = "pynumero"
            DoE_args["scale_nominal_param_value"] = False

            doe_obj = DesignOfExperiments(**DoE_args)
            # Build polynomial factorial results and draw the linear-scale 2D plots.
            doe_obj.compute_FIM_full_factorial(
                design_ranges={"x1": [0, 5, 2], "x2": [0, 5, 2]}
            )
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
            experiment=[exp],
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
        expected_det = 11.0
        expected_pseudo_trace = 7.0
        expected_trace = 7.0 / 11.0

        doe_det = self._make_template_doe("determinant")
        self.assertAlmostEqual(
            doe_det._evaluate_objective_from_fim(fim), expected_det, places=10
        )

        doe_ptr = self._make_template_doe("pseudo_trace")
        self.assertAlmostEqual(
            doe_ptr._evaluate_objective_from_fim(fim), expected_pseudo_trace, places=10
        )

        doe_tr = self._make_template_doe("trace")
        self.assertAlmostEqual(
            doe_tr._evaluate_objective_from_fim(fim), expected_trace, places=10
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
        # Force the positive-definiteness check down the jitter path and verify
        # the matrix passed into Cholesky includes the expected diagonal shift.
        doe = self._make_template_doe("determinant")

        from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS

        original_solve = doe.solver.solve
        original_eigvals = np.linalg.eigvals
        original_cholesky = np.linalg.cholesky
        solve_count = {"n": 0}
        eig_call_count = {"n": 0}
        cholesky_inputs = []

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

        def _eigvals_force_jitter(*args, **kwargs):
            eig_call_count["n"] += 1
            if eig_call_count["n"] == 1:
                return np.array([-1.0, 1.0])
            return original_eigvals(*args, **kwargs)

        def _capture_cholesky(mat, *args, **kwargs):
            cholesky_inputs.append(np.array(mat, copy=True))
            return original_cholesky(mat, *args, **kwargs)

        with patch(
            "pyomo.contrib.doe.doe.np.linalg.eigvals", side_effect=_eigvals_force_jitter
        ) as eigvals_mock:
            with patch(
                "pyomo.contrib.doe.doe.np.linalg.cholesky",
                side_effect=_capture_cholesky,
            ):
                with patch.object(
                    doe.solver, "solve", side_effect=_solve_first_real_then_mock
                ):
                    doe.optimize_experiments(n_exp=1)

        scenario = doe.model.param_scenario_blocks[0]
        total_fim = np.array(
            _optimize_experiments_param_scenario(doe.results)["total_fim"]
        )
        expected_cholesky_input = total_fim + _SMALL_TOLERANCE_DEFINITENESS * np.eye(
            total_fim.shape[0]
        )
        param_names = list(scenario.exp_blocks[0].parameter_names)
        L_vals = np.array(
            [
                [pyo.value(scenario.obj_cons.L[p, q]) for q in param_names]
                for p in param_names
            ]
        )

        self.assertEqual(doe.results["optimization_solve"]["status"], "ok")
        self.assertGreaterEqual(eigvals_mock.call_count, 1)
        self.assertTrue(cholesky_inputs)
        self.assertTrue(
            any(
                np.allclose(chol_arg, expected_cholesky_input, atol=1e-12)
                for chol_arg in cholesky_inputs
            )
        )
        self.assertTrue(
            np.allclose(L_vals @ L_vals.T, expected_cholesky_input, atol=1e-8)
        )

    def test_lhs_initialize_experiments_matches_bruteforce_combo(self):
        # Compare the helper's chosen initial design directly against an
        # independent brute-force scorer so this test isolates LHS selection
        # instead of the later NLP solve and result-payload bookkeeping.
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

        doe = self._make_template_doe("pseudo_trace")
        self._build_template_model_for_multi_experiment(doe, n_exp=n_exp)
        actual_points, lhs_diag = doe._lhs_initialize_experiments(
            lhs_n_samples=lhs_n_samples,
            lhs_seed=lhs_seed,
            n_exp=n_exp,
            lhs_parallel=False,
            lhs_combo_parallel=False,
        )

        actual_points_norm = sorted(tuple(np.round(p, 8)) for p in actual_points)
        expected_points_norm = sorted(tuple(np.round(p, 8)) for p in expected_points)
        self.assertEqual(actual_points_norm, expected_points_norm)
        self.assertAlmostEqual(lhs_diag["best_obj"], best_obj, places=12)

    def test_optimize_experiments_is_reentrant_on_same_object(self):
        # Tests that optimize_experiments() can be run repeatedly on one DoE object.
        doe = self._make_template_doe("pseudo_trace")
        doe.optimize_experiments(n_exp=1)
        first_design = _optimize_experiments_param_scenario(doe.results)["experiments"][
            0
        ]["design"]
        first_build_time = doe.results["timing"]["build_time_s"]

        doe.optimize_experiments(n_exp=1)
        second_design = _optimize_experiments_param_scenario(doe.results)[
            "experiments"
        ][0]["design"]

        self.assertEqual(len(first_design), len(second_design))
        self.assertIn("timing", doe.results)
        self.assertGreater(doe.results["timing"]["build_time_s"], 0.0)
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
        # Track every submitted future so we can later verify which pending
        # evaluations were explicitly cancelled when the deadline is exceeded.
        created_futures = []

        class _FakeFuture:
            def __init__(self, idx):
                self.idx = idx
                self.cancelled = False

            def result(self):
                # Return a deterministic candidate-FIM payload keyed by submit order.
                x = float(self.idx + 1.0)
                return self.idx, np.array([[x, 0.0], [0.0, x + 1.0]])

            def cancel(self):
                self.cancelled = True
                return True

        class _FakeExecutor:
            # Record submitted work but do not execute it asynchronously; the
            # test controls completion order through the patched wait() call.
            def __init__(self, max_workers=None):
                self.created = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, idx_pt):
                fut = _FakeFuture(idx_pt[0])
                self.created.append(fut)
                created_futures.append(fut)
                return fut

        def _fake_wait(pending, timeout=None, return_when=None):
            # Pretend exactly one future finishes per wait cycle and leave the
            # rest pending so timeout handling has something to cancel.
            done = {min(pending, key=lambda fut: fut.idx)}
            still_pending = set(pending) - done
            return done, still_pending

        perf_counter_calls = {"n": 0}

        def _fake_perf_counter():
            # Keep the clock at t=0 long enough to fill the initial pending
            # queue, then jump past the deadline so the code cancels leftovers.
            perf_counter_calls["n"] += 1
            if perf_counter_calls["n"] <= 6:
                return 0.0
            return 1.0

        with patch("pyomo.contrib.doe.doe._cf.ThreadPoolExecutor", _FakeExecutor):
            with patch("pyomo.contrib.doe.doe._cf.wait", side_effect=_fake_wait):
                with patch(
                    "pyomo.contrib.doe.doe.time.perf_counter",
                    side_effect=_fake_perf_counter,
                ):
                    points, diag = doe._lhs_initialize_experiments(
                        lhs_n_samples=20,
                        lhs_seed=5,
                        n_exp=2,
                        lhs_parallel=True,
                        lhs_n_workers=2,
                        lhs_combo_parallel=False,
                        lhs_max_wall_clock_time=0.5,
                    )

        self.assertEqual(len(points), 2)
        self.assertTrue(diag["timed_out"])
        # With 2 workers, the implementation allows up to 4 pending candidate
        # FIM futures before waiting for one to complete.
        self.assertEqual(len(created_futures), 4)
        # Our fake scheduler completes futures 0 and 1, so timeout should only
        # cancel the still-pending futures from the tail of that first batch.
        self.assertEqual(sum(fut.cancelled for fut in created_futures), 2)
        self.assertEqual([fut.idx for fut in created_futures if fut.cancelled], [2, 3])

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
        created_futures = []

        class _FakeFuture:
            def __init__(self, idx, payload):
                self.idx = idx
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
                fut = _FakeFuture(idx, payload)
                self.created.append(fut)
                created_futures.append(fut)
                return fut

        def _fake_wait(pending, return_when=None):
            done = {min(pending, key=lambda fut: fut.idx)}
            still_pending = set(pending) - done
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
        self.assertEqual(len(created_futures), 3)
        self.assertEqual(sum(fut.cancelled for fut in created_futures), 2)
        self.assertEqual([fut.idx for fut in created_futures if fut.cancelled], [1, 2])

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
        # Tests determinant-objective optimization against known expected design/metric values.
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
            experiment=exp_list,
            objective_option="determinant",
            step=1e-2,
            solver=solver,
        )
        doe.optimize_experiments()

        scenario = _optimize_experiments_param_scenario(doe.results)
        got_hours = sorted(exp["design"][0] for exp in scenario["experiments"])
        expected_hours = [1.9321985035514362, 9.999999685577139]

        self.assertStructuredAlmostEqual(got_hours, expected_hours, abstol=1e-3)
        self.assertAlmostEqual(
            scenario["quality_metrics"]["log10_d_opt"], 6.028152580313302, places=3
        )

    def test_optimize_experiments_trace_expected_values(self):
        # Tests trace-objective optimization against known expected design/metric values.
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
            experiment=exp_list,
            objective_option="trace",
            step=1e-2,
            solver=solver,
            prior_FIM=prior_FIM,
        )
        doe.optimize_experiments()

        scenario = _optimize_experiments_param_scenario(doe.results)
        got_hours = sorted(exp["design"][0] for exp in scenario["experiments"])
        expected_hours = [10.0, 10.0]

        self.assertStructuredAlmostEqual(got_hours, expected_hours, abstol=1e-3)
        self.assertAlmostEqual(
            scenario["quality_metrics"]["log10_a_opt"], -2.2347, places=3
        )

    def test_optimize_experiments_prior_fim_aggregation_non_lhs_template_mode(self):
        # Tests that total FIM equals sum(experiment FIMs) + prior in template mode.
        prior_fim = np.array([[2.0, 0.1], [0.1, 1.5]])
        doe = self._make_template_doe("pseudo_trace")
        doe.prior_FIM = prior_fim.copy()

        doe.optimize_experiments(n_exp=2, init_method=None)

        scenario = _optimize_experiments_param_scenario(doe.results)
        total_fim = np.array(scenario["total_fim"])
        exp_fim_sum = sum(
            (np.array(exp_data["fim"]) for exp_data in scenario["experiments"]),
            np.zeros_like(total_fim),
        )
        stored_prior = np.array(doe.results["problem"]["prior_fim"])

        self.assertTrue(np.allclose(total_fim, exp_fim_sum + prior_fim, atol=1e-6))
        self.assertTrue(np.allclose(total_fim, exp_fim_sum + stored_prior, atol=1e-6))

    def test_optimize_experiments_prior_fim_aggregation_non_lhs_user_initialized_mode(
        self,
    ):
        # Tests that total FIM aggregation with prior is correct in user-initialized mode.
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
            experiment=exp_list,
            objective_option="pseudo_trace",
            step=1e-2,
            solver=solver,
            prior_FIM=prior_fim,
        )
        doe.optimize_experiments(init_method=None)

        scenario = _optimize_experiments_param_scenario(doe.results)
        total_fim = np.array(scenario["total_fim"])
        exp_fim_sum = sum(
            (np.array(exp_data["fim"]) for exp_data in scenario["experiments"]),
            np.zeros_like(total_fim),
        )
        stored_prior = np.array(doe.results["problem"]["prior_fim"])

        self.assertTrue(np.allclose(total_fim, exp_fim_sum + prior_fim, atol=1e-6))
        self.assertTrue(np.allclose(total_fim, exp_fim_sum + stored_prior, atol=1e-6))

    def test_optimize_experiments_safe_metric_failure_sets_nan(self):
        # Tests that metric-computation failures are captured as NaN with a warning.
        doe = self._make_template_doe("pseudo_trace")
        with patch(
            "pyomo.contrib.doe.doe.np.linalg.inv", side_effect=RuntimeError("boom")
        ):
            with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as log_cm:
                doe.optimize_experiments(n_exp=1)

        scenario = _optimize_experiments_param_scenario(doe.results)
        self.assertTrue(np.isnan(scenario["quality_metrics"]["log10_a_opt"]))
        self.assertTrue(
            any("failed to compute log10 A-opt" in msg for msg in log_cm.output)
        )

    def test_optimize_experiments_non_cholesky_determinant_initialization(self):
        # Tests determinant initialization correctness when Cholesky formulation is disabled.
        exp = RooneyBieglerMultiExperiment(hour=2.0, y=10.0)
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        doe = DesignOfExperiments(
            experiment=[exp],
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
        total_fim = np.array(
            _optimize_experiments_param_scenario(doe.results)["total_fim"]
        )
        expected_det = np.linalg.det(total_fim)
        self.assertAlmostEqual(
            pyo.value(scenario_block.obj_cons.determinant), expected_det, places=6
        )


if __name__ == "__main__":
    unittest.main()
