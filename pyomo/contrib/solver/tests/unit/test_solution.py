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

from pyomo.common import unittest
from pyomo.contrib.solver.common.solution_loader import (
    SolutionLoaderBase,
    PersistentSolutionLoader,
)


class TestSolutionLoaderBase(unittest.TestCase):
    def test_member_list(self):
        expected_list = [
            'load_vars',
            'get_vars',
            'get_duals',
            'get_reduced_costs',
            'load_import_suffixes',
            'get_number_of_solutions',
            'get_solution_ids',
            'load_solution',
        ]
        method_list = [
            method
            for method in dir(SolutionLoaderBase)
            if method.startswith('_') is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_solution_loader_base(self):
        self.instance = SolutionLoaderBase()
        with self.assertRaises(NotImplementedError):
            self.instance.get_vars()
        self.assertEqual(self.instance.get_duals(), NotImplemented)
        self.assertEqual(self.instance.get_reduced_costs(), NotImplemented)


class TestSolSolutionLoader(unittest.TestCase):
    # I am currently unsure how to test this further because it relies heavily on
    # SolFileData and NLWriterInfo
    def test_member_list(self):
        expected_list = [
            'load_vars',
            'get_vars',
            'get_duals',
            'get_reduced_costs',
            'load_import_suffixes',
            'get_number_of_solutions',
            'get_solution_ids',
            'load_solution',
        ]
        method_list = [
            method
            for method in dir(SolutionLoaderBase)
            if method.startswith('_') is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))


class TestPersistentSolutionLoader(unittest.TestCase):
    def test_member_list(self):
        expected_list = [
            'load_vars',
            'get_vars',
            'get_duals',
            'get_reduced_costs',
            'invalidate',
            'load_import_suffixes',
            'get_number_of_solutions',
            'get_solution_ids',
            'load_solution',
        ]
        method_list = [
            method
            for method in dir(PersistentSolutionLoader)
            if method.startswith('_') is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_initialization(self):
        # Realistically, a solver object should be passed into this.
        # However, it works with a string. It'll just error loudly if you
        # try to run get_primals, etc.
        self.instance = PersistentSolutionLoader('ipopt', None)
        self.assertTrue(self.instance._valid)
        self.assertEqual(self.instance._solver, 'ipopt')

    def test_invalid(self):
        self.instance = PersistentSolutionLoader('ipopt', None)
        self.instance.invalidate()
        with self.assertRaises(RuntimeError):
            self.instance.get_vars()
