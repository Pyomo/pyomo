#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import itertools
from unittest import skipUnless
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.triangulations import (
    get_j1_triangulation,
    get_incremental_simplex_ordering,
    get_incremental_simplex_ordering_assume_connected_by_n_face,
)

class TestTriangulations(unittest.TestCase):

    def test_J1_small(self):
        points = [
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2],
        ]
        triangulation = get_j1_triangulation(points, 2)
        self.assertEqual(triangulation.simplices,
        {
            0: [[0, 0], [0, 1], [1, 1]],
            1: [[0, 1], [0, 2], [1, 1]],
            2: [[1, 1], [2, 0], [2, 1]],
            3: [[1, 1], [2, 1], [2, 2]],
            4: [[0, 0], [1, 0], [1, 1]],
            5: [[0, 2], [1, 1], [1, 2]],
            6: [[1, 0], [1, 1], [2, 0]],
            7: [[1, 1], [1, 2], [2, 2]],
        })
    
    # check that the points_map functionality does what it should
    def test_J1_small_offset(self):
        points = [
            [0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
            [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
            [2.5, 0.5], [2.5, 1.5], [2.5, 2.5],
        ]
        triangulation = get_j1_triangulation(points, 2)
        self.assertEqual(triangulation.simplices,
        {
            0: [[0.5, 0.5], [0.5, 1.5], [1.5, 1.5]],
            1: [[0.5, 1.5], [0.5, 2.5], [1.5, 1.5]],
            2: [[1.5, 1.5], [2.5, 0.5], [2.5, 1.5]],
            3: [[1.5, 1.5], [2.5, 1.5], [2.5, 2.5]],
            4: [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]],
            5: [[0.5, 2.5], [1.5, 1.5], [1.5, 2.5]],
            6: [[1.5, 0.5], [1.5, 1.5], [2.5, 0.5]],
            7: [[1.5, 1.5], [1.5, 2.5], [2.5, 2.5]],
        })
    
    def test_J1_small_ordering(self):
        points = [
            [0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
            [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
            [2.5, 0.5], [2.5, 1.5], [2.5, 2.5],
        ]
        triangulation = get_j1_triangulation(points, 2)
        reordered_simplices = get_incremental_simplex_ordering(triangulation.simplices)
        for idx, first_simplex in reordered_simplices.items():
            if idx != len(triangulation.points) - 1:
                second_simplex = reordered_simplices[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")
    
    def test_J1_medium_ordering(self):
        points = list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6]))
        triangulation = get_j1_triangulation(points, 2)
        reordered_simplices = get_incremental_simplex_ordering(triangulation.simplices)
        for idx, first_simplex in reordered_simplices.items():
            if idx != len(triangulation.points) - 1:
                second_simplex = reordered_simplices[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")

    def test_J1_medium_ordering_alt(self):
        points = list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6]))
        triangulation = get_j1_triangulation(points, 2)
        reordered_simplices = get_incremental_simplex_ordering_assume_connected_by_n_face(triangulation.simplices, 1)
        for idx, first_simplex in reordered_simplices.items():
            if idx != len(triangulation.points) - 1:
                second_simplex = reordered_simplices[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")

    def test_J1_medium_ordering_3d(self):
        points = list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6], [-5, -1, 0.2, 3, 10]))
        triangulation = get_j1_triangulation(points, 3)
        reordered_simplices = get_incremental_simplex_ordering_assume_connected_by_n_face(triangulation.simplices, 2)
        for idx, first_simplex in reordered_simplices.items():
            if idx != len(triangulation.points) - 1:
                second_simplex = reordered_simplices[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")
