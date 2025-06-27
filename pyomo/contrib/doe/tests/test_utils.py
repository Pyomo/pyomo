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
    snake_traversal_grid_sampling,
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
        self.assertEqual(det_FIM, det_expected)
        self.assertEqual(trace_FIM, trace_expected)
        self.assertTrue(np.allclose(E_vals, E_vals_expected))
        self.assertTrue(np.allclose(E_vecs, E_vecs_expected))
        self.assertEqual(D_opt, D_opt_expected)
        self.assertEqual(A_opt, A_opt_expected)
        self.assertEqual(E_opt, E_opt_expected)
        self.assertEqual(ME_opt, ME_opt_expected)

    def test_FIM_eigenvalue_warning(self):
        # Create a matrix with an imaginary component large enough
        # to trigger the warning
        FIM = np.array([[6, 5j], [5j, 7]])
        with self.assertLogs("pyomo.contrib.doe.utils", level="WARNING") as cm:
            compute_FIM_metrics(FIM)
            expected_warning = (
                "Eigenvalue has imaginary component greater than "
                + f"{_SMALL_TOLERANCE_IMG}, contact developers if this issue persists."
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
        self.assertEqual(fim_metrics["Determinant of FIM"], det_expected)
        self.assertEqual(fim_metrics["Trace of FIM"], trace_expected)
        self.assertTrue(np.allclose(fim_metrics["Eigenvalues"], E_vals_expected))
        self.assertTrue(np.allclose(fim_metrics["Eigenvectors"], E_vecs_expected))
        self.assertEqual(fim_metrics["log10(D-Optimality)"], D_opt_expected)
        self.assertEqual(fim_metrics["log10(A-Optimality)"], A_opt_expected)
        self.assertEqual(fim_metrics["log10(E-Optimality)"], E_opt_expected)
        self.assertEqual(fim_metrics["log10(Modified E-Optimality)"], ME_opt_expected)

    def test_snake_traversal_grid_sampling_errors(self):
        # Test the error handling with lists
        list_2d_bad = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError) as cm:
            list(snake_traversal_grid_sampling(list_2d_bad))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D. Got shape (2, 3).",
        )

        list_2d_wrong_shape_bad = [[1, 2, 3], [4, 5, 6, 7]]
        with self.assertRaises(ValueError) as cm:
            list(snake_traversal_grid_sampling(list_2d_wrong_shape_bad))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D array-like.",
        )

        # Test the error handling with tuples
        tuple_2d_bad = ((1, 2, 3), (4, 5, 6))
        with self.assertRaises(ValueError) as cm:
            list(snake_traversal_grid_sampling(tuple_2d_bad))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D. Got shape (2, 3).",
        )

        tuple_2d_wrong_shape_bad = ((1, 2, 3), (4, 5, 6, 7))
        with self.assertRaises(ValueError) as cm:
            list(snake_traversal_grid_sampling(tuple_2d_wrong_shape_bad))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D array-like.",
        )

        # Test the error handling with numpy arrays
        array_2d_bad = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError) as cm:
            list(snake_traversal_grid_sampling(array_2d_bad))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D. Got shape (2, 3).",
        )

    def test_snake_traversal_grid_sampling_values(self):
        # Test with lists
        # Test with a single list
        list1 = [1, 2, 3]
        result_list1 = list(snake_traversal_grid_sampling(list1))
        expected_list1 = [(1,), (2,), (3,)]
        self.assertEqual(result_list1, expected_list1)

        # Test with two lists
        list2 = [4, 5, 6]
        result_list2 = list(snake_traversal_grid_sampling(list1, list2))
        expected_list2 = [
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 6),
            (2, 5),
            (2, 4),
            (3, 4),
            (3, 5),
            (3, 6),
        ]
        self.assertEqual(result_list2, expected_list2)

        # Test with three lists
        list3 = [7, 8]
        result_list3 = list(snake_traversal_grid_sampling(list1, list2, list3))
        expected_list3 = [
            (1, 4, 7),
            (1, 4, 8),
            (1, 5, 8),
            (1, 5, 7),
            (1, 6, 7),
            (1, 6, 8),
            (2, 6, 8),
            (2, 6, 7),
            (2, 5, 7),
            (2, 5, 8),
            (2, 4, 8),
            (2, 4, 7),
            (3, 4, 7),
            (3, 4, 8),
            (3, 5, 8),
            (3, 5, 7),
            (3, 6, 7),
            (3, 6, 8),
        ]
        self.assertEqual(result_list3, expected_list3)

        # Test with tuples
        tuple1 = (1, 2, 3)
        result_tuple1 = list(snake_traversal_grid_sampling(tuple1))
        tuple2 = (4, 5, 6)
        result_tuple2 = list(snake_traversal_grid_sampling(tuple1, tuple2))
        tuple3 = (7, 8)
        result_tuple3 = list(snake_traversal_grid_sampling(tuple1, tuple2, tuple3))
        self.assertEqual(result_tuple1, expected_list1)
        self.assertEqual(result_tuple2, expected_list2)
        self.assertEqual(result_tuple3, expected_list3)

        # Test with numpy arrays
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])
        array3 = np.array([7, 8])
        result_array1 = list(snake_traversal_grid_sampling(array1))
        result_array2 = list(snake_traversal_grid_sampling(array1, array2))
        result_array3 = list(snake_traversal_grid_sampling(array1, array2, array3))
        self.assertEqual(result_array1, expected_list1)
        self.assertEqual(result_array2, expected_list2)
        self.assertEqual(result_array3, expected_list3)

        # Test with mixed types(List, Tuple, numpy array)
        result_mixed = list(snake_traversal_grid_sampling(list1, tuple2, array3))
        self.assertEqual(result_mixed, expected_list3)


if __name__ == "__main__":
    unittest.main()
