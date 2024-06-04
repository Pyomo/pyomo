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
    get_Gn_hamiltonian,
)
from math import factorial
import itertools

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
    
    def test_J1_2d_ordering_0(self):
        points = list(itertools.product([0, 1, 2], [1, 2.4, 3]))
        ordered_triangulation = get_j1_triangulation(points, 2, ordered=True).simplices
        self.assertEqual(len(ordered_triangulation), 8)
        for idx, first_simplex in ordered_triangulation.items():
            if idx != len(ordered_triangulation) - 1:
                second_simplex = ordered_triangulation[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")

    def test_J1_2d_ordering_1(self):
        points = list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6]))
        ordered_triangulation = get_j1_triangulation(points, 2, ordered=True).simplices
        self.assertEqual(len(ordered_triangulation), 32)
        for idx, first_simplex in ordered_triangulation.items():
            if idx != len(ordered_triangulation) - 1:
                second_simplex = ordered_triangulation[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")

    def test_J1_2d_ordering_2(self):
        points = list(itertools.product([0, 1, 2, 4, 5, 6.3, 7.1], [1, 2.4, 3, 5, 6, 9.1, 10]))
        ordered_triangulation = get_j1_triangulation(points, 2, ordered=True).simplices
        self.assertEqual(len(ordered_triangulation), 72)
        for idx, first_simplex in ordered_triangulation.items():
            if idx != len(ordered_triangulation) - 1:
                second_simplex = ordered_triangulation[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")

    def test_J1_2d_ordering_3(self):
        points = list(itertools.product([0, 1, 2, 4, 5, 6.3, 7.1, 7.2, 7.3], [1, 2.4, 3, 5, 6, 9.1, 10, 11, 12]))
        ordered_triangulation = get_j1_triangulation(points, 2, ordered=True).simplices
        self.assertEqual(len(ordered_triangulation), 128)
        for idx, first_simplex in ordered_triangulation.items():
            if idx != len(ordered_triangulation) - 1:
                second_simplex = ordered_triangulation[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")
    
    def check_Gn_hamiltonian_path(self, n, start_permutation, target_symbol, last):
        path = get_Gn_hamiltonian(n, start_permutation, target_symbol, last)
        self.assertEqual(len(path), factorial(n))
        self.assertEqual(path[0], start_permutation)
        if last:
            self.assertEqual(path[-1][-1], target_symbol)
        else:
            self.assertEqual(path[-1][0], target_symbol)
        for pi in itertools.permutations(range(1, n + 1), n):
            self.assertTrue(tuple(pi) in path)
        for i in range(len(path) - 1):
            diff_indices = [j for j in range(n) if path[i][j] != path[i + 1][j]]
            self.assertEqual(len(diff_indices), 2)
            self.assertEqual(diff_indices[0], diff_indices[1] - 1)
            self.assertEqual(path[i][diff_indices[0]], path[i + 1][diff_indices[1]])
            self.assertEqual(path[i][diff_indices[1]], path[i + 1][diff_indices[0]])

    def test_Gn_hamiltonian_paths(self):
        # each of the base cases
        self.check_Gn_hamiltonian_path(4, (1, 2, 3, 4), 1, False)
        self.check_Gn_hamiltonian_path(4, (1, 2, 3, 4), 2, False)
        self.check_Gn_hamiltonian_path(4, (1, 2, 3, 4), 3, False)
        self.check_Gn_hamiltonian_path(4, (1, 2, 3, 4), 4, False)
        # some variants with start permutations and/or last
        self.check_Gn_hamiltonian_path(4, (3, 4, 1, 2), 2, False)
        self.check_Gn_hamiltonian_path(4, (1, 3, 2, 4), 3, True)
        self.check_Gn_hamiltonian_path(4, (1, 4, 2, 3), 4, True)
        self.check_Gn_hamiltonian_path(4, (1, 2, 3, 4), 2, True)
        # some recursive cases
        self.check_Gn_hamiltonian_path(5, (1, 2, 3, 4, 5), 1, False)
        self.check_Gn_hamiltonian_path(5, (1, 2, 3, 4, 5), 3, False)
        self.check_Gn_hamiltonian_path(5, (1, 2, 3, 4, 5), 5, False)
        self.check_Gn_hamiltonian_path(5, (1, 2, 4, 3, 5), 5, True)
        self.check_Gn_hamiltonian_path(6, (6, 1, 2, 4, 3, 5), 5, True)
        self.check_Gn_hamiltonian_path(6, (6, 1, 2, 4, 3, 5), 5, False)
        self.check_Gn_hamiltonian_path(7, (1, 2, 3, 4, 5, 6, 7), 7, False)
        
