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
from pyomo.contrib.doe.utils import generate_snake_zigzag_pattern


@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestUtilsFIM(unittest.TestCase):
    def test_generate_snake_zigzag_pattern_errors(self):
        # Test the error handling with lists
        list_2d = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError) as cm:
            list(generate_snake_zigzag_pattern(list_2d))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D. Got shape (2, 3).",
        )

        list_2d_wrong_shape = [[1, 2, 3], [4, 5, 6, 7]]
        with self.assertRaises(ValueError) as cm:
            list(generate_snake_zigzag_pattern(list_2d_wrong_shape))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D array-like.",
        )

        # Test the error handling with tuples
        tuple_2d = ((1, 2, 3), (4, 5, 6))
        with self.assertRaises(ValueError) as cm:
            list(generate_snake_zigzag_pattern(tuple_2d))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D. Got shape (2, 3).",
        )

        tuple_2d_wrong_shape = ((1, 2, 3), (4, 5, 6, 7))
        with self.assertRaises(ValueError) as cm:
            list(generate_snake_zigzag_pattern(tuple_2d_wrong_shape))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D array-like.",
        )

        # Test the error handling with numpy arrays
        array_2d = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError) as cm:
            list(generate_snake_zigzag_pattern(array_2d))
        self.assertEqual(
            str(cm.exception),
            "Argument at position 0 is not 1D. Got shape (2, 3).",
        )

    def test_generate_snake_zigzag_pattern_values(self):
        # Test with lists
        # Test with a single list
        list1 = [1, 2, 3]
        result_list1 = list(generate_snake_zigzag_pattern(list1))
        expected_list1 = [(1,), (2,), (3,)]
        self.assertEqual(result_list1, expected_list1)

        # Test with two lists
        list2 = [4, 5, 6]
        result_list2 = list(generate_snake_zigzag_pattern(list1, list2))
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
        result_list3 = list(generate_snake_zigzag_pattern(list1, list2, list3))
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
        result_tuple1 = list(generate_snake_zigzag_pattern(tuple1))
        tuple2 = (4, 5, 6)
        result_tuple2 = list(generate_snake_zigzag_pattern(tuple1, tuple2))
        tuple3 = (7, 8)
        result_tuple3 = list(generate_snake_zigzag_pattern(tuple1, tuple2, tuple3))
        self.assertEqual(result_tuple1, expected_list1)
        self.assertEqual(result_tuple2, expected_list2)
        self.assertEqual(result_tuple3, expected_list3)

        # Test with numpy arrays
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])
        array3 = np.array([7, 8])
        result_array1 = list(generate_snake_zigzag_pattern(array1))
        result_array2 = list(generate_snake_zigzag_pattern(array1, array2))
        result_array3 = list(generate_snake_zigzag_pattern(array1, array2, array3))
        self.assertEqual(result_array1, expected_list1)
        self.assertEqual(result_array2, expected_list2)
        self.assertEqual(result_array3, expected_list3)

        # Test with mixed types(List, Tuple, numpy array)
        result_mixed = list(generate_snake_zigzag_pattern(list1, tuple2, array3))
        self.assertEqual(result_mixed, expected_list3)


if __name__ == "__main__":
    unittest.main()
