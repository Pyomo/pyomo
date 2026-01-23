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
from pyomo.common.dependencies import numpy as np, numpy_available

import pyomo.common.unittest as unittest
from pyomo.contrib.doe.utils import (
    check_FIM,
    compute_FIM_metrics,
    get_FIM_metrics,
    _SMALL_TOLERANCE_DEFINITENESS,
    _SMALL_TOLERANCE_SYMMETRY,
    _SMALL_TOLERANCE_IMG,
)


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

    def test_compute_FIM_metrics(self):
        # Create a sample Fisher Information Matrix (FIM)
        FIM = np.array([[10, 2], [2, 3]])

        det_FIM, trace_FIM, E_vals, E_vecs, D_opt, A_opt, E_opt, ME_opt = (
            compute_FIM_metrics(FIM)
        )

        # expected results
        det_expected = 26.000000000000004
        D_opt_expected = 1.414973347970818

        trace_expected = 13
        A_opt_expected = 1.1139433523068367

        E_vals_expected = np.array([10.53112887, 2.46887113])
        E_vecs_expected = np.array(
            [[0.96649965, -0.25666794], [0.25666794, 0.96649965]]
        )
        E_opt_expected = 0.3924984205140895

        ME_opt_expected = 0.6299765069426388

        # Test results
        self.assertAlmostEqual(det_FIM, det_expected)
        self.assertAlmostEqual(trace_FIM, trace_expected)
        self.assertTrue(np.allclose(E_vals, E_vals_expected))
        self.assertTrue(np.allclose(E_vecs, E_vecs_expected))
        self.assertAlmostEqual(D_opt, D_opt_expected)
        self.assertAlmostEqual(A_opt, A_opt_expected)
        self.assertAlmostEqual(E_opt, E_opt_expected)
        self.assertAlmostEqual(ME_opt, ME_opt_expected)

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
        FIM = np.array([[10, 2], [2, 3]])
        fim_metrics = get_FIM_metrics(FIM)

        # expected results
        det_expected = 26.000000000000004
        D_opt_expected = 1.414973347970818

        trace_expected = 13
        A_opt_expected = 1.1139433523068367

        E_vals_expected = np.array([10.53112887, 2.46887113])
        E_vecs_expected = np.array(
            [[0.96649965, -0.25666794], [0.25666794, 0.96649965]]
        )
        E_opt_expected = 0.3924984205140895

        ME_opt_expected = 0.6299765069426388

        # Test results
        self.assertAlmostEqual(fim_metrics["Determinant of FIM"], det_expected)
        self.assertAlmostEqual(fim_metrics["Trace of FIM"], trace_expected)
        self.assertTrue(np.allclose(fim_metrics["Eigenvalues"], E_vals_expected))
        self.assertTrue(np.allclose(fim_metrics["Eigenvectors"], E_vecs_expected))
        self.assertAlmostEqual(fim_metrics["log10(D-Optimality)"], D_opt_expected)
        self.assertAlmostEqual(fim_metrics["log10(A-Optimality)"], A_opt_expected)
        self.assertAlmostEqual(fim_metrics["log10(E-Optimality)"], E_opt_expected)
        self.assertAlmostEqual(
            fim_metrics["log10(Modified E-Optimality)"], ME_opt_expected
        )


if __name__ == "__main__":
    unittest.main()
