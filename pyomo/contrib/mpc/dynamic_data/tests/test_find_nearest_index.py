#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
import pyomo.common.unittest as unittest
import pytest

from idaes.apps.nmpc.dynamic_data.find_nearest_index import (
    find_nearest_index,
)


@pytest.mark.component
class TestFindNearestIndex(unittest.TestCase):

    def test_two_points(self):
        array = [0, 5]

        i = find_nearest_index(array, 1)
        self.assertEqual(i, 0)
        i = find_nearest_index(array, 1, tolerance=0.5)
        self.assertEqual(i, None)

        i = find_nearest_index(array, -0.01, tolerance=0.1)
        self.assertEqual(i, 0)
        i = find_nearest_index(array, -0.01, tolerance=0.001)
        self.assertEqual(i, None)

        i = find_nearest_index(array, 6, tolerance=2)
        self.assertEqual(i, 1)
        i = find_nearest_index(array, 6, tolerance=1)
        self.assertEqual(i, 1)

        # This test relies on the behavior for tiebreaks
        i = find_nearest_index(array, 2.5)
        self.assertEqual(i, 0)

    def test_array_with_floats(self):
        array = []
        for i in range(5):
            i0 = float(i)
            i1 = round((i + 0.15) * 1e4)/1e4
            i2 = round((i + 0.64) * 1e4)/1e4
            array.extend([i, i1, i2])
        array.append(5.0)

        i = find_nearest_index(array, 1.01, tolerance=0.1)
        self.assertEqual(i, 3)
        i = find_nearest_index(array, 1.01, tolerance=0.001)
        self.assertEqual(i, None)

        i = find_nearest_index(array, 3.5)
        self.assertEqual(i, 11)
        i = find_nearest_index(array, 3.5, tolerance=0.1)
        self.assertEqual(i, None)

        i = find_nearest_index(array, -1)
        self.assertEqual(i, 0)
        i = find_nearest_index(array, -1, tolerance=1)
        self.assertEqual(i, 0)

        i = find_nearest_index(array, 5.5)
        self.assertEqual(i, 15)
        i = find_nearest_index(array, 5.5, tolerance=0.49)
        self.assertEqual(i, None)

        i = find_nearest_index(array, 2.64, tolerance=1e-8)
        self.assertEqual(i, 8)
        i = find_nearest_index(array, 2.64, tolerance=0)
        self.assertEqual(i, 8)

        i = find_nearest_index(array, 5, tolerance=0)
        self.assertEqual(i, 15)

        i = find_nearest_index(array, 0, tolerance=0)
        self.assertEqual(i, 0)


if __name__ == "__main__":
    unittest.main()
