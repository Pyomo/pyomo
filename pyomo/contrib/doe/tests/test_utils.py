# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
import copy
import json
import os.path

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)
from pyomo.common.fileutils import this_file_dir

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.doe.utils import (
    ExperimentGradients,
    check_FIM,
    compute_FIM_metrics,
    get_FIM_metrics,
    _SMALL_TOLERANCE_DEFINITENESS,
    _SMALL_TOLERANCE_SYMMETRY,
    _SMALL_TOLERANCE_IMG,
)

if scipy_available:
    from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
    from pyomo.contrib.doe.examples.polynomial import PolynomialExperiment
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
        RooneyBieglerExperiment,
    )
from pyomo.opt import SolverFactory

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)

data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}

ipopt_available = SolverFactory("ipopt").available()



@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestUtilsFIM(unittest.TestCase):
    """Test the check_FIM() from utils.py."""

    def test_check_FIM_valid(self):
        """Test case where the FIM is valid (square, positive definite, symmetric)."""
        FIM = np.array([[4, 1], [1, 3]])
        try:
            check_FIM(FIM)
        except ValueError as e:
            self.fail(f"Unexpected error: {e}")

    def test_check_FIM_non_square(self):
        """Test case where the FIM is not square."""
        FIM = np.array([[4, 1], [1, 3], [2, 1]])
        with self.assertRaisesRegex(ValueError, "FIM must be a square matrix"):
            check_FIM(FIM)

    def test_check_FIM_non_positive_definite(self):
        """Test case where the FIM is not positive definite."""
        FIM = np.array([[1, 0], [0, -2]])
        with self.assertRaisesRegex(
            ValueError,
            "FIM provided is not positive definite. It has one or more negative "
            + r"eigenvalue\(s\) less than -{:.1e}".format(
                _SMALL_TOLERANCE_DEFINITENESS
            ),
        ):
            check_FIM(FIM)

    def test_check_FIM_non_symmetric(self):
        """Test case where the FIM is not symmetric."""
        FIM = np.array([[4, 1], [0, 3]])
        with self.assertRaisesRegex(
            ValueError,
            "FIM provided is not symmetric using absolute tolerance {}".format(
                _SMALL_TOLERANCE_SYMMETRY
            ),
        ):
            check_FIM(FIM)

    """Test the compute_FIM_metrics() from utils.py."""

    ### Helper methods for test cases
    # Sample FIM for testing
    def _get_test_fim(self):
        """Helper method returning test FIM matrix."""
        return np.array([[10, 2], [2, 3]])

    # Expected results for the test FIM
    def _get_expected_fim_results(self):
        """Helper method returning expected FIM computation results."""
        return {
            'det': 26.000000000000004,
            'D_opt': 1.414973347970818,
            'trace_cov': 0.5,
            'A_opt': -0.3010299956639812,
            'trace_FIM': 13,
            'pseudo_A_opt': 1.1139433523068367,
            'E_vals': np.array([10.53112887, 2.46887113]),
            'E_vecs': np.array([[0.96649965, -0.25666794], [0.25666794, 0.96649965]]),
            'E_opt': 0.3924984205140895,
            'ME_opt': 0.6299765069426388,
        }

    def test_compute_FIM_metrics(self):
        # Create a sample Fisher Information Matrix (FIM)
        FIM = self._get_test_fim()
        # expected results
        expected = self._get_expected_fim_results()

        (
            det_FIM,
            trace_cov,
            trace_FIM,
            E_vals,
            E_vecs,
            D_opt,
            A_opt,
            pseudo_A_opt,
            E_opt,
            ME_opt,
        ) = compute_FIM_metrics(FIM)

        # Test results
        self.assertAlmostEqual(det_FIM, expected['det'])
        self.assertAlmostEqual(trace_cov, expected['trace_cov'])
        self.assertAlmostEqual(trace_FIM, expected['trace_FIM'])
        self.assertTrue(np.allclose(E_vals, expected['E_vals']))
        self.assertTrue(np.allclose(E_vecs, expected['E_vecs']))
        self.assertAlmostEqual(D_opt, expected['D_opt'])
        self.assertAlmostEqual(A_opt, expected['A_opt'])
        self.assertAlmostEqual(pseudo_A_opt, expected['pseudo_A_opt'])
        self.assertAlmostEqual(E_opt, expected['E_opt'])
        self.assertAlmostEqual(ME_opt, expected['ME_opt'])

    def test_FIM_eigenvalue_warning(self):
        # Create a matrix with an imaginary component large enough
        # to trigger the warning
        FIM = np.array([[6, 5j], [5j, 7]])
        with self.assertLogs("pyomo.contrib.doe.utils", level="WARNING") as cm:
            compute_FIM_metrics(FIM)
            expected_warning = (
                "Eigenvalue has imaginary component greater than "
                + f"{_SMALL_TOLERANCE_IMG}, contact the developers if this issue "
                + "persists."
            )
            self.assertIn(expected_warning, cm.output[0])

    """Test the get_FIM_metrics() from utils.py."""

    def test_get_FIM_metrics(self):
        # Create a sample Fisher Information Matrix (FIM)
        FIM = self._get_test_fim()
        # expected results
        expected = self._get_expected_fim_results()
        fim_metrics = get_FIM_metrics(FIM)

        # Test results
        self.assertAlmostEqual(fim_metrics["Determinant of FIM"], expected['det'])
        self.assertAlmostEqual(fim_metrics["Trace of cov"], expected['trace_cov'])
        self.assertAlmostEqual(fim_metrics["Trace of FIM"], expected['trace_FIM'])
        self.assertTrue(np.allclose(fim_metrics["Eigenvalues"], expected['E_vals']))
        self.assertTrue(np.allclose(fim_metrics["Eigenvectors"], expected['E_vecs']))
        self.assertAlmostEqual(fim_metrics["log10(D-Optimality)"], expected['D_opt'])
        self.assertAlmostEqual(fim_metrics["log10(A-Optimality)"], expected['A_opt'])
        self.assertAlmostEqual(
            fim_metrics["log10(Pseudo A-Optimality)"], expected['pseudo_A_opt']
        )
        self.assertAlmostEqual(fim_metrics["log10(E-Optimality)"], expected['E_opt'])
        self.assertAlmostEqual(
            fim_metrics["log10(Modified E-Optimality)"], expected['ME_opt']
        )


@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestExperimentGradients(unittest.TestCase):
    """Validate symbolic and automatic differentiation helpers."""

    def _assert_symbolic_and_automatic_jacobians_agree(
        self, model, atol=1e-8, rtol=1e-8
    ):
        """Check that symbolic and automatic Jacobian tables agree entry by entry."""
        experiment_gradients = ExperimentGradients(model, symbolic=True, automatic=True)

        self.assertEqual(
            set(experiment_gradients.jac_dict_sd), set(experiment_gradients.jac_dict_ad)
        )
        for key in experiment_gradients.jac_dict_sd:
            self.assertTrue(
                np.isclose(
                    pyo.value(experiment_gradients.jac_dict_sd[key]),
                    pyo.value(experiment_gradients.jac_dict_ad[key]),
                    atol=atol,
                    rtol=rtol,
                ),
                msg=f"Mismatch at Jacobian entry {key}",
            )
        return experiment_gradients

    def _get_rooney_biegler_experiment(
        self, hour=5.0, y=15.6, asymptote=15.0, rate_constant=0.5, measure_error=0.1
    ):
        """Build a Rooney-Biegler experiment for gradient validation."""
        data = pd.DataFrame(data=[[hour, y]], columns=["hour", "y"])
        return RooneyBieglerExperiment(
            data=data.iloc[0],
            theta={"asymptote": asymptote, "rate_constant": rate_constant},
            measure_error=measure_error,
        )

    def _get_reactor_experiment(self, ca0=5.0, temperature_offset=0.0, nfe=5, ncp=2):
        """Build a lightly perturbed reactor experiment for gradient validation."""
        reactor_data = copy.deepcopy(data_ex)
        reactor_data["CA0"] = ca0
        reactor_data["control_points"] = {
            t: value + temperature_offset
            for t, value in reactor_data["control_points"].items()
        }
        return ReactorExperiment(data=reactor_data, nfe=nfe, ncp=ncp)

    def _initialize_reactor_model(self, model):
        """Solve the reactor model once to populate state values for AD checks."""
        for v in model.experiment_inputs.keys():
            v.fix()
        for v in model.unknown_parameters.keys():
            v.fix()

        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        results = solver.solve(model, tee=False)

        self.assertEqual(str(results.solver.status).lower(), "ok")
        return model

    def _get_expected_polynomial_gradient(self):
        """Return the exact gradient of the polynomial output at the test point."""
        return np.array([[2.0, 3.0, 6.0, 1.0]])

    def _get_expected_polynomial_fim_with_prior(self):
        """Return a positive-definite polynomial FIM used for metric regression."""
        gradient = self._get_expected_polynomial_gradient().ravel()
        return np.outer(gradient, gradient) + np.eye(4)

    def _evaluate_polynomial_output(self, a, b, c, d, x1=2.0, x2=3.0):
        """Evaluate the scalar polynomial model at the test point."""
        return a * x1 + b * x2 + c * x1 * x2 + d

    def test_polynomial_gradients_match_expected(self):
        """Check polynomial output sensitivities against analytic values."""
        experiment = PolynomialExperiment()
        model = experiment.get_labeled_model()

        experiment_gradients = ExperimentGradients(model, symbolic=True, automatic=True)
        jacobian = (
            experiment_gradients.compute_gradient_outputs_wrt_unknown_parameters()
        )

        expected = self._get_expected_polynomial_gradient()

        self.assertEqual(jacobian.shape, expected.shape)
        self.assertTrue(np.allclose(jacobian, expected))

    def test_polynomial_symbolic_and_automatic_jacobians_agree(self):
        """Ensure symbolic and automatic Jacobian entries agree exactly."""
        experiment = PolynomialExperiment()
        model = experiment.get_labeled_model()

        self._assert_symbolic_and_automatic_jacobians_agree(model)

    def test_polynomial_symbolic_matches_manual_central_difference(self):
        """Check symbolic sensitivities against a manual central-difference estimate."""
        experiment = PolynomialExperiment()
        model = experiment.get_labeled_model()
        experiment_gradients = ExperimentGradients(model, symbolic=True, automatic=True)

        symbolic = (
            experiment_gradients.compute_gradient_outputs_wrt_unknown_parameters()
            .ravel()
            .astype(float)
        )
        base_values = {"a": 2.0, "b": -1.0, "c": 0.5, "d": -1.0}
        step = 1e-6
        finite_difference = []
        for parameter in ("a", "b", "c", "d"):
            forward_values = dict(base_values)
            backward_values = dict(base_values)
            forward_values[parameter] += step
            backward_values[parameter] -= step
            forward = self._evaluate_polynomial_output(**forward_values)
            backward = self._evaluate_polynomial_output(**backward_values)
            finite_difference.append((forward - backward) / (2 * step))

        self.assertTrue(np.allclose(symbolic, finite_difference, atol=1e-7, rtol=1e-7))

    def test_polynomial_automatic_only_still_sets_both_jacobians(self):
        """Check that both Jacobian maps are prepared in the unified setup path."""
        experiment = PolynomialExperiment()
        model = experiment.get_labeled_model()

        experiment_gradients = ExperimentGradients(
            model, symbolic=False, automatic=True
        )

        self.assertIsNotNone(experiment_gradients.jac_dict_sd)
        self.assertIsNotNone(experiment_gradients.jac_dict_ad)

    def test_polynomial_metric_helpers_match_numpy(self):
        """Check utility metrics on a polynomial-derived positive-definite FIM."""
        FIM = self._get_expected_polynomial_fim_with_prior()
        fim_metrics = get_FIM_metrics(FIM)
        (
            det_FIM,
            trace_cov,
            trace_FIM,
            E_vals,
            E_vecs,
            D_opt,
            A_opt,
            pseudo_A_opt,
            E_opt,
            ME_opt,
        ) = compute_FIM_metrics(FIM)

        self.assertAlmostEqual(det_FIM, np.linalg.det(FIM))
        self.assertAlmostEqual(trace_cov, np.trace(np.linalg.inv(FIM)))
        self.assertAlmostEqual(trace_FIM, np.trace(FIM))

        expected_eigs, _expected_vecs = np.linalg.eigh(FIM)
        self.assertTrue(np.allclose(np.sort(E_vals), np.sort(expected_eigs)))
        self.assertEqual(E_vecs.shape, FIM.shape)

        self.assertAlmostEqual(D_opt, np.log10(np.linalg.det(FIM)))
        self.assertAlmostEqual(A_opt, np.log10(np.trace(np.linalg.inv(FIM))))
        self.assertAlmostEqual(pseudo_A_opt, np.log10(np.trace(FIM)))
        self.assertAlmostEqual(E_opt, np.log10(np.min(expected_eigs)))
        self.assertAlmostEqual(
            ME_opt, np.log10(np.max(expected_eigs) / np.min(expected_eigs))
        )

        self.assertAlmostEqual(fim_metrics["Determinant of FIM"], np.linalg.det(FIM))
        self.assertAlmostEqual(
            fim_metrics["Trace of cov"], np.trace(np.linalg.inv(FIM))
        )
        self.assertAlmostEqual(fim_metrics["Trace of FIM"], np.trace(FIM))

    def test_polynomial_symbolic_only_still_sets_both_jacobians(self):
        """Check that symbolic-only requests still initialize both Jacobian maps."""
        experiment = PolynomialExperiment()
        model = experiment.get_labeled_model()

        experiment_gradients = ExperimentGradients(
            model, symbolic=True, automatic=False
        )

        self.assertIsNotNone(experiment_gradients.jac_dict_sd)
        self.assertIsNotNone(experiment_gradients.jac_dict_ad)

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_symbolic_and_automatic_jacobians_agree(self):
        """Check Rooney-Biegler Jacobians from symbolic and automatic differentiation."""
        experiment = self._get_rooney_biegler_experiment()
        model = experiment.get_labeled_model()

        self._assert_symbolic_and_automatic_jacobians_agree(model)

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_rooney_biegler_gradients_match_closed_form(self):
        """Check Rooney-Biegler sensitivities against the closed-form derivatives."""
        hour = 7.0
        asymptote = 14.0
        rate_constant = 0.4
        experiment = self._get_rooney_biegler_experiment(
            hour=hour, y=19.8, asymptote=asymptote, rate_constant=rate_constant
        )
        model = experiment.get_labeled_model()
        experiment_gradients = self._assert_symbolic_and_automatic_jacobians_agree(
            model
        )

        jacobian = (
            experiment_gradients.compute_gradient_outputs_wrt_unknown_parameters()
        )
        expected = np.array(
            [
                [
                    1.0 - np.exp(-rate_constant * hour),
                    asymptote * hour * np.exp(-rate_constant * hour),
                ]
            ]
        )

        self.assertEqual(jacobian.shape, expected.shape)
        self.assertTrue(np.allclose(jacobian, expected, atol=1e-7, rtol=1e-7))

    @unittest.skipIf(not scipy_available, "scipy is not available")
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_symbolic_and_automatic_jacobians_agree(self):
        """Check reactor Jacobians from symbolic and automatic differentiation."""
        experiment = self._get_reactor_experiment()
        model = self._initialize_reactor_model(experiment.get_labeled_model())

        experiment_gradients = self._assert_symbolic_and_automatic_jacobians_agree(
            model, atol=1e-6, rtol=1e-6
        )

        self.assertGreater(len(experiment_gradients.jac_dict_sd), 0)
        self.assertEqual(
            len(experiment_gradients.jac_dict_sd), len(experiment_gradients.jac_dict_ad)
        )

    @unittest.skipIf(not scipy_available, "scipy is not available")
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_symbolic_and_automatic_jacobians_agree_at_perturbed_point(self):
        """Check reactor Jacobian agreement at a perturbed operating point."""
        experiment = self._get_reactor_experiment(ca0=4.0, temperature_offset=25.0)
        model = self._initialize_reactor_model(experiment.get_labeled_model())

        experiment_gradients = self._assert_symbolic_and_automatic_jacobians_agree(
            model, atol=1e-6, rtol=1e-6
        )

        self.assertGreater(len(experiment_gradients.jac_dict_sd), 0)
        self.assertEqual(
            len(experiment_gradients.jac_dict_sd), len(experiment_gradients.jac_dict_ad)
        )


if __name__ == "__main__":
    unittest.main()
