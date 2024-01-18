#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import unittest
from pyomo.contrib.solver import solution


class TestPersistentSolverBase(unittest.TestCase):
    def test_abstract_member_list(self):
        expected_list = ['get_primals']
        member_list = list(solution.SolutionLoaderBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    @unittest.mock.patch.multiple(
        solution.SolutionLoaderBase, __abstractmethods__=set()
    )
    def test_solution_loader_base(self):
        self.instance = solution.SolutionLoaderBase()
        self.assertEqual(self.instance.get_primals(), None)
        with self.assertRaises(NotImplementedError):
            self.instance.get_duals()
        with self.assertRaises(NotImplementedError):
            self.instance.get_reduced_costs()
