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
from pyomo.contrib.doe.doe import (
    _SMALL_TOLERANCE_IMG,
    _compute_FIM_metrics,
)


@unittest.skipIf(not numpy_available, "Numpy is not available")
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
