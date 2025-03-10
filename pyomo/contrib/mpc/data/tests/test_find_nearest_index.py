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

import pyomo.common.unittest as unittest
import pytest

from pyomo.contrib.mpc.data.find_nearest_index import (
    find_nearest_index,
    find_nearest_interval_index,
)


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
            i1 = round((i + 0.15) * 1e4) / 1e4
            i2 = round((i + 0.64) * 1e4) / 1e4
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


class TestFindNearestIntervalIndex(unittest.TestCase):
    def test_find_interval(self):
        intervals = [(0.0, 0.1), (0.1, 0.5), (0.7, 1.0)]
        target = 0.05
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)

        target = 0.099
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)

        target = 0.1
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)

        target = 0.1
        idx = find_nearest_interval_index(intervals, target, prefer_left=False)
        self.assertEqual(idx, 1)

        target = 0.55
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 1)

        target = 0.60
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 1)

        target = 0.6999
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 2)

        target = 1.0
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 2)

        target = -0.1
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)

        target = 1.1
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 2)

    def test_find_interval_tolerance(self):
        intervals = [(0.0, 0.1), (0.1, 0.5), (0.7, 1.0)]

        target = 0.501
        idx = find_nearest_interval_index(intervals, target, tolerance=None)
        self.assertEqual(idx, 1)

        idx = find_nearest_interval_index(intervals, target, tolerance=1e-5)
        self.assertEqual(idx, None)

        idx = find_nearest_interval_index(intervals, target, tolerance=1e-2)
        self.assertEqual(idx, 1)

        target = 1.001
        idx = find_nearest_interval_index(intervals, target, tolerance=1e-2)
        self.assertEqual(idx, 2)

        #
        # Behavior when distance between target and nearest interval "equals"
        # the tolerance is not well-defined. Here the computed distance may
        # not be exactly 1e-3 due to roundoff error.
        #
        # idx = find_nearest_interval_index(intervals, target, tolerance=1e-3)
        # self.assertEqual(idx, None)

        idx = find_nearest_interval_index(intervals, target, tolerance=1e-4)
        self.assertEqual(idx, None)

    def test_find_interval_with_tolerance_on_boundary(self):
        # Our target is on the boundary between two intervals.
        intervals = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)]
        target = 0.1001
        idx = find_nearest_interval_index(
            intervals, target, tolerance=None, prefer_left=True
        )
        self.assertEqual(idx, 1)

        # target != 0.1 (the interval boundary) within tolerance. We are
        # within interval 1.
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-5, prefer_left=True
        )
        self.assertEqual(idx, 1)
        # This is true even if we prefer the right interval
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-5, prefer_left=False
        )
        self.assertEqual(idx, 1)

        # target == 0.1 within tolerance. We are on the boundary, and
        # should return the "preferred" interval.
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=True
        )
        self.assertEqual(idx, 0)
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=False
        )
        self.assertEqual(idx, 1)

        target = 0.4999
        # We are not equal to boundary (0.5) within tolerance
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-5, prefer_left=True
        )
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-5, prefer_left=False
        )
        self.assertEqual(idx, 1)

        # We are equal to boundary within tolerance
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=True
        )
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=False
        )
        self.assertEqual(idx, 2)

    def test_find_interval_with_tolerance_singleton(self):
        intervals = [(0.0, 0.1), (0.1, 0.1), (0.5, 0.5), (0.5, 1.0)]

        target = 0.1001
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=True
        )
        self.assertEqual(idx, 0)
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=False
        )
        self.assertEqual(idx, 1)

        target = 0.0999
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=True
        )
        self.assertEqual(idx, 0)
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=False
        )
        self.assertEqual(idx, 1)

        target = 0.4999
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=True
        )
        self.assertEqual(idx, 2)
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=False
        )
        self.assertEqual(idx, 3)

        target = 0.5001
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=True
        )
        self.assertEqual(idx, 2)
        idx = find_nearest_interval_index(
            intervals, target, tolerance=1e-3, prefer_left=False
        )
        self.assertEqual(idx, 3)


if __name__ == "__main__":
    unittest.main()
