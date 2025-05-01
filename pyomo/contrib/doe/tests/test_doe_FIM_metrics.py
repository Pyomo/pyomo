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
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
)

import pyomo.common.unittest as unittest
from pyomo.contrib.doe import DesignOfExperiments

# Not need? from pyomo.contrib.doe.tests import doe_test_example
from pyomo.contrib.doe.doe import (
    _SMALL_TOLERANCE_IMG,
    _SMALL_TOLERANCE_DEFINITENESS,
    _SMALL_TOLERANCE_SYMMETRY,
    _compute_FIM_metrics,
)


@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestDesignOfExperimentsCheckFIM(unittest.TestCase):
    """Test the check_FIM method of the DesignOfExperiments class."""

    def test_check_FIM_valid(self):
        """Test case where the FIM is valid (square, positive definite, symmetric)."""
        FIM = np.array([[4, 1], [1, 3]])
        try:
            # Call the static method directly
            DesignOfExperiments._check_FIM(FIM)
        except ValueError as e:
            self.fail(f"Unexpected error: {e}")

    def test_check_FIM_non_square(self):
        """Test case where the FIM is not square."""
        FIM = np.array([[4, 1], [1, 3], [2, 1]])
        with self.assertRaisesRegex(ValueError, "FIM must be a square matrix"):
            DesignOfExperiments._check_FIM(FIM)

    def test_check_FIM_non_positive_definite(self):
        """Test case where the FIM is not positive definite."""
        FIM = np.array([[1, 0], [0, -2]])
        with self.assertRaisesRegex(
            ValueError,
            r"FIM provided is not positive definite. It has one or more negative eigenvalue\(s\) less than -{:.1e}".format(
                _SMALL_TOLERANCE_DEFINITENESS
            ),
            # r"FIM provided is not positive definite. .*",
        ):
            DesignOfExperiments._check_FIM(FIM)

    def test_check_FIM_non_symmetric(self):
        """Test case where the FIM is not symmetric."""
        FIM = np.array([[4, 1], [0, 3]])
        with self.assertRaisesRegex(
            ValueError,
            "FIM provided is not symmetric using absolute tolerance {}".format(
                _SMALL_TOLERANCE_SYMMETRY
            ),
        ):
            DesignOfExperiments._check_FIM(FIM)


class TestComputeFIMMetrics(unittest.TestCase):
    def test_compute_FIM_metrics(self):
        # Create a sample Fisher Information Matrix (FIM)
        FIM = np.array([[10, 2], [2, 3]])

        det_FIM, trace_FIM, E_vals, E_vecs, D_opt, A_opt, E_opt, ME_opt = (
            _compute_FIM_metrics(FIM)
        )

        # expected results
        det_expected = np.linalg.det(FIM)
        D_opt_expected = np.log10(det_expected)

        trace_expected = np.trace(FIM)
        A_opt_expected = np.log10(trace_expected)

        E_vals_expected, E_vecs_expected = np.linalg.eig(FIM)
        min_eigval = np.min(E_vals_expected.real)
        if min_eigval <= 0:
            E_opt_expected = np.nan
        else:
            E_opt_expected = np.log10(min_eigval)

        cond_expected = np.linalg.cond(FIM)

        ME_opt_expected = np.log10(cond_expected)

        # Test results
        self.assertEqual(det_FIM, det_expected)
        self.assertEqual(trace_FIM, trace_expected)
        self.assertTrue(np.allclose(E_vals, E_vals_expected))
        self.assertTrue(np.allclose(E_vecs, E_vecs_expected))
        self.assertEqual(D_opt, D_opt_expected)
        self.assertEqual(A_opt, A_opt_expected)
        self.assertEqual(E_opt, E_opt_expected)
        self.assertEqual(ME_opt, ME_opt_expected)


class TestFIMWarning(unittest.TestCase):
    def test_FIM_eigenvalue_warning(self):
        # Create a matrix with an imaginary component large enough to trigger the warning
        FIM = np.array([[6, 5j], [5j, 7]])
        with self.assertLogs("pyomo.contrib.doe", level="WARNING") as cm:
            _compute_FIM_metrics(FIM)
            expected_warning = f"Eigenvalue has imaginary component greater than {_SMALL_TOLERANCE_IMG}, contact developers if this issue persists."
            self.assertIn(expected_warning, cm.output[0])


if __name__ == "__main__":
    unittest.main()
