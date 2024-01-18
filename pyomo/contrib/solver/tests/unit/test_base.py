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
from pyomo.contrib.solver import base


class TestSolverBase(unittest.TestCase):
    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_solver_base(self):
        self.instance = base.SolverBase()
        self.assertFalse(self.instance.is_persistent())
        self.assertEqual(self.instance.version(), None)
        self.assertEqual(self.instance.CONFIG, self.instance.config)
        self.assertEqual(self.instance.solve(None), None)
        self.assertEqual(self.instance.available(), None)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_solver_availability(self):
        self.instance = base.SolverBase()
        self.instance.Availability._value_ = 1
        self.assertTrue(self.instance.Availability.__bool__(self.instance.Availability))
        self.instance.Availability._value_ = -1
        self.assertFalse(
            self.instance.Availability.__bool__(self.instance.Availability)
        )


class TestPersistentSolverBase(unittest.TestCase):
    def test_abstract_member_list(self):
        expected_list = [
            'remove_params',
            'version',
            'update_variables',
            'remove_variables',
            'add_constraints',
            '_get_primals',
            'set_instance',
            'set_objective',
            'update_params',
            'remove_block',
            'add_block',
            'available',
            'add_params',
            'remove_constraints',
            'add_variables',
            'solve',
        ]
        member_list = list(base.PersistentSolverBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    @unittest.mock.patch.multiple(base.PersistentSolverBase, __abstractmethods__=set())
    def test_persistent_solver_base(self):
        self.instance = base.PersistentSolverBase()
        self.assertTrue(self.instance.is_persistent())
        self.assertEqual(self.instance.set_instance(None), None)
        self.assertEqual(self.instance.add_variables(None), None)
        self.assertEqual(self.instance.add_params(None), None)
        self.assertEqual(self.instance.add_constraints(None), None)
        self.assertEqual(self.instance.add_block(None), None)
        self.assertEqual(self.instance.remove_variables(None), None)
        self.assertEqual(self.instance.remove_params(None), None)
        self.assertEqual(self.instance.remove_constraints(None), None)
        self.assertEqual(self.instance.remove_block(None), None)
        self.assertEqual(self.instance.set_objective(None), None)
        self.assertEqual(self.instance.update_variables(None), None)
        self.assertEqual(self.instance.update_params(), None)

        with self.assertRaises(NotImplementedError):
            self.instance._get_primals()

        with self.assertRaises(NotImplementedError):
            self.instance._get_duals()

        with self.assertRaises(NotImplementedError):
            self.instance._get_reduced_costs()
