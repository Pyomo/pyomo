# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
from pyomo.common.dependencies import numpy as np, numpy_available

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe.utils import (
    ExperimentGradients,
    check_FIM,
    compute_FIM_metrics,
    get_FIM_metrics,
    _SMALL_TOLERANCE_DEFINITENESS,
    _SMALL_TOLERANCE_SYMMETRY,
    _SMALL_TOLERANCE_IMG,
)


class PolynomialExperiment(Experiment):
    """A small polynomial experiment used to validate symbolic gradients."""

    def __init__(self):
        self.model = None

    def get_labeled_model(self):
        """Build and label the experiment model on first access."""
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model

    def create_model(self):
        m = self.model = pyo.ConcreteModel()
        m.x1 = pyo.Var(bounds=(-5, 5), initialize=2)
        m.x2 = pyo.Var(bounds=(-5, 5), initialize=3)
        m.a = pyo.Var(bounds=(-5, 5), initialize=2)
        m.b = pyo.Var(bounds=(-5, 5), initialize=-1)
        m.c = pyo.Var(bounds=(-5, 5), initialize=0.5)
        m.d = pyo.Var(bounds=(-5, 5), initialize=-1)
        m.y = pyo.Var(initialize=0)

        @m.Constraint()
        def output_equation(m):
            return m.y == m.a * m.x1 + m.b * m.x2 + m.c * m.x1 * m.x2 + m.d

    def finalize_model(self):
        """No additional model finalization is needed for this example."""
        pass

    def label_experiment(self):
        """Attach the standard DoE suffixes to the polynomial model."""
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y] = None

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y] = 1

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.x1] = None
        m.experiment_inputs[m.x2] = None

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.a, m.b, m.c, m.d])


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

        experiment_gradients = ExperimentGradients(model, symbolic=True, automatic=True)

        self.assertEqual(
            set(experiment_gradients.jac_dict_sd), set(experiment_gradients.jac_dict_ad)
        )
        for key in experiment_gradients.jac_dict_sd:
            self.assertAlmostEqual(
                pyo.value(experiment_gradients.jac_dict_sd[key]),
                pyo.value(experiment_gradients.jac_dict_ad[key]),
            )

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


if __name__ == "__main__":
    unittest.main()
