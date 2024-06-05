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
    get_unordered_j1_triangulation,
    get_ordered_j1_triangulation,
    get_Gn_hamiltonian,
    get_grid_hamiltonian,
)
from math import factorial
import itertools

class TestTriangulations(unittest.TestCase):

    # check basic functionality for the unordered j1 triangulation.
    def test_J1_small(self):
        points = [
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2],
        ]
        triangulation = get_unordered_j1_triangulation(points, 2)
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
        triangulation = get_unordered_j1_triangulation(points, 2)
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

    def check_J1_ordered(self, points, num_points, dim):
        ordered_triangulation = get_ordered_j1_triangulation(points, dim).simplices
        #print(ordered_triangulation)
        self.assertEqual(len(ordered_triangulation), factorial(dim) * (num_points - 1) ** dim)
        for idx, first_simplex in ordered_triangulation.items():
            if idx != len(ordered_triangulation) - 1:
                second_simplex = ordered_triangulation[idx + 1]
                # test property (2) which also guarantees property (1)
                self.assertEqual(first_simplex[-1], second_simplex[0], msg="Last and first vertices of adjacent simplices did not match")
                # The way I am constructing these, they should always share an (n-1)-face.
                # Check that too for good measure.
                count = 0
                for pt in first_simplex:
                    if pt in second_simplex:
                        count += 1
                #print(f"first_simplex={first_simplex}; second_simplex={second_simplex}")
                self.assertEqual(count, dim) # (n-1)-face has n points
                #if count != dim:
                #    print(f"error: count {count} was not the correct {dim}")
    
    def test_J1_ordered_2d(self):
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2], [1, 2.4, 3])),
            3,
            2,
        )
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6])),
            5,
            2,
        )
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5, 6.3, 7.1], [1, 2.4, 3, 5, 6, 9.1, 10])),
            7,
            2,
        )
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5, 6.3, 7.1, 7.2, 7.3], [1, 2.4, 3, 5, 6, 9.1, 10, 11, 12])),
            9,
            2,
        )
    
    def test_J1_ordered_3d(self):
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2], [1, 2.4, 3], [2, 3, 4])),
            3,
            3,
        )
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6], [-1, 0, 1, 2, 3])),
            5,
            3,
        )
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5, 6, 7], [1, 2.4, 3, 5, 6, 6.5, 7], [-1, 0, 1, 2, 3, 4, 5])),
            7,
            3,
        )
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5, 6, 7, 8, 9], [1, 2.4, 3, 5, 6, 6.5, 7, 8, 9], [-1, 0, 1, 2, 3, 4, 5, 6, 7])),
            9,
            3,
        )
    
    def test_J1_ordered_4d_and_above(self):
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6], [-1, 0, 1, 2, 3], [1, 2, 3, 4, 5])),
            5,
            4,
        )
        self.check_J1_ordered(
            list(itertools.product([0, 1, 2, 4, 5], [1, 2.4, 3, 5, 6], [-1, 0, 1, 2, 3], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6])),
            5,
            5,
        )

    
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
    
    def check_grid_hamiltonian(self, dim, length):
        path = get_grid_hamiltonian(dim, length)
        self.assertEqual(len(path), length ** dim)
        for x in itertools.product(range(length), repeat=dim):
            self.assertTrue(list(x) in path)
        for i in range(len(path) - 1):
            diff_indices = [j for j in range(dim) if path[i][j] != path[i + 1][j]]
            self.assertEqual(len(diff_indices), 1)
            self.assertEqual(abs(path[i][diff_indices[0]] - path[i + 1][diff_indices[0]]), 1)
    
    def test_grid_hamiltonian_paths(self):
        self.check_grid_hamiltonian(1, 5)
        self.check_grid_hamiltonian(2, 5)
        self.check_grid_hamiltonian(2, 8)
        self.check_grid_hamiltonian(3, 5)
        self.check_grid_hamiltonian(4, 3)
        
        
