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

import os

from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base


class TestSolverBase(unittest.TestCase):
    def test_abstract_member_list(self):
        expected_list = ['solve', 'available', 'version']
        member_list = list(base.SolverBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    def test_class_method_list(self):
        expected_list = [
            'Availability',
            'CONFIG',
            'available',
            'is_persistent',
            'solve',
            'version',
        ]
        method_list = [
            method for method in dir(base.SolverBase) if method.startswith('_') is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_init(self):
        self.instance = base.SolverBase()
        self.assertFalse(self.instance.is_persistent())
        self.assertEqual(self.instance.version(), None)
        self.assertEqual(self.instance.name, 'solverbase')
        self.assertEqual(self.instance.CONFIG, self.instance.config)
        self.assertEqual(self.instance.solve(None), None)
        self.assertEqual(self.instance.available(), None)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_context_manager(self):
        with base.SolverBase() as self.instance:
            self.assertFalse(self.instance.is_persistent())
            self.assertEqual(self.instance.version(), None)
            self.assertEqual(self.instance.name, 'solverbase')
            self.assertEqual(self.instance.CONFIG, self.instance.config)
            self.assertEqual(self.instance.solve(None), None)
            self.assertEqual(self.instance.available(), None)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_config_kwds(self):
        self.instance = base.SolverBase(tee=True)
        self.assertTrue(self.instance.config.tee)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_solver_availability(self):
        self.instance = base.SolverBase()
        self.instance.Availability._value_ = 1
        self.assertTrue(self.instance.Availability.__bool__(self.instance.Availability))
        self.instance.Availability._value_ = -1
        self.assertFalse(
            self.instance.Availability.__bool__(self.instance.Availability)
        )

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_custom_solver_name(self):
        self.instance = base.SolverBase(name='my_unique_name')
        self.assertEqual(self.instance.name, 'my_unique_name')


class TestPersistentSolverBase(unittest.TestCase):
    def test_abstract_member_list(self):
        expected_list = [
            'remove_parameters',
            'version',
            'update_variables',
            'remove_variables',
            'add_constraints',
            '_get_primals',
            'set_instance',
            'set_objective',
            'update_parameters',
            'remove_block',
            'add_block',
            'available',
            'add_parameters',
            'remove_constraints',
            'add_variables',
            'solve',
        ]
        member_list = list(base.PersistentSolverBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    def test_class_method_list(self):
        expected_list = [
            'Availability',
            'CONFIG',
            '_get_duals',
            '_get_primals',
            '_get_reduced_costs',
            '_load_vars',
            'add_block',
            'add_constraints',
            'add_parameters',
            'add_variables',
            'available',
            'is_persistent',
            'remove_block',
            'remove_constraints',
            'remove_parameters',
            'remove_variables',
            'set_instance',
            'set_objective',
            'solve',
            'update_parameters',
            'update_variables',
            'version',
        ]
        method_list = [
            method
            for method in dir(base.PersistentSolverBase)
            if (method.startswith('__') or method.startswith('_abc')) is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    @unittest.mock.patch.multiple(base.PersistentSolverBase, __abstractmethods__=set())
    def test_init(self):
        self.instance = base.PersistentSolverBase()
        self.assertTrue(self.instance.is_persistent())
        self.assertEqual(self.instance.set_instance(None), None)
        self.assertEqual(self.instance.add_variables(None), None)
        self.assertEqual(self.instance.add_parameters(None), None)
        self.assertEqual(self.instance.add_constraints(None), None)
        self.assertEqual(self.instance.add_block(None), None)
        self.assertEqual(self.instance.remove_variables(None), None)
        self.assertEqual(self.instance.remove_parameters(None), None)
        self.assertEqual(self.instance.remove_constraints(None), None)
        self.assertEqual(self.instance.remove_block(None), None)
        self.assertEqual(self.instance.set_objective(None), None)
        self.assertEqual(self.instance.update_variables(None), None)
        self.assertEqual(self.instance.update_parameters(), None)

        with self.assertRaises(NotImplementedError):
            self.instance._get_primals()

        with self.assertRaises(NotImplementedError):
            self.instance._get_duals()

        with self.assertRaises(NotImplementedError):
            self.instance._get_reduced_costs()

    @unittest.mock.patch.multiple(base.PersistentSolverBase, __abstractmethods__=set())
    def test_context_manager(self):
        with base.PersistentSolverBase() as self.instance:
            self.assertTrue(self.instance.is_persistent())
            self.assertEqual(self.instance.set_instance(None), None)
            self.assertEqual(self.instance.add_variables(None), None)
            self.assertEqual(self.instance.add_parameters(None), None)
            self.assertEqual(self.instance.add_constraints(None), None)
            self.assertEqual(self.instance.add_block(None), None)
            self.assertEqual(self.instance.remove_variables(None), None)
            self.assertEqual(self.instance.remove_parameters(None), None)
            self.assertEqual(self.instance.remove_constraints(None), None)
            self.assertEqual(self.instance.remove_block(None), None)
            self.assertEqual(self.instance.set_objective(None), None)
            self.assertEqual(self.instance.update_variables(None), None)
            self.assertEqual(self.instance.update_parameters(), None)


class TestLegacySolverWrapper(unittest.TestCase):
    def test_class_method_list(self):
        expected_list = ['available', 'license_is_valid', 'solve']
        method_list = [
            method
            for method in dir(base.LegacySolverWrapper)
            if method.startswith('_') is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_context_manager(self):
        with base.LegacySolverWrapper() as instance:
            with self.assertRaises(AttributeError):
                instance.available()

    def test_map_config(self):
        # Create a fake/empty config structure that can be added to an empty
        # instance of LegacySolverWrapper
        self.config = ConfigDict(implicit=True)
        self.config.declare(
            'solver_options',
            ConfigDict(implicit=True, description="Options to pass to the solver."),
        )
        instance = base.LegacySolverWrapper()
        instance.config = self.config
        instance._map_config(
            True, False, False, 20, True, False, None, None, None, False, None, None
        )
        self.assertTrue(instance.config.tee)
        self.assertFalse(instance.config.load_solutions)
        self.assertEqual(instance.config.time_limit, 20)
        # Report timing shouldn't be created because it no longer exists
        with self.assertRaises(AttributeError):
            print(instance.config.report_timing)
        # Keepfiles should not be created because we did not declare keepfiles on
        # the original config
        with self.assertRaises(AttributeError):
            print(instance.config.keepfiles)
        # We haven't implemented solver_io, suffixes, or logfile
        with self.assertRaises(NotImplementedError):
            instance._map_config(
                False,
                False,
                False,
                20,
                False,
                False,
                None,
                None,
                '/path/to/bogus/file',
                False,
                None,
                None,
            )
        with self.assertRaises(NotImplementedError):
            instance._map_config(
                False,
                False,
                False,
                20,
                False,
                False,
                None,
                '/path/to/bogus/file',
                None,
                False,
                None,
                None,
            )
        with self.assertRaises(NotImplementedError):
            instance._map_config(
                False,
                False,
                False,
                20,
                False,
                False,
                '/path/to/bogus/file',
                None,
                None,
                False,
                None,
                None,
            )
        # If they ask for keepfiles, we redirect them to working_dir
        instance._map_config(
            False, False, False, 20, False, False, None, None, None, True, None, None
        )
        self.assertEqual(instance.config.working_dir, os.getcwd())
        with self.assertRaises(AttributeError):
            print(instance.config.keepfiles)

    def test_map_results(self):
        # Unclear how to test this
        pass

    def test_solution_handler(self):
        # Unclear how to test this
        pass
