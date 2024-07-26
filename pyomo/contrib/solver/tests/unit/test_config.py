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

from pyomo.common import unittest
from pyomo.contrib.solver.config import (
    SolverConfig,
    BranchAndBoundConfig,
    AutoUpdateConfig,
    PersistentSolverConfig,
)


class TestSolverConfig(unittest.TestCase):
    def test_interface_default_instantiation(self):
        config = SolverConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)

    def test_interface_custom_instantiation(self):
        config = SolverConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertFalse(config.time_limit)
        config.time_limit = 1.0
        self.assertEqual(config.time_limit, 1.0)
        self.assertIsInstance(config.time_limit, float)


class TestBranchAndBoundConfig(unittest.TestCase):
    def test_interface_default_instantiation(self):
        config = BranchAndBoundConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.rel_gap)
        self.assertIsNone(config.abs_gap)

    def test_interface_custom_instantiation(self):
        config = BranchAndBoundConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertFalse(config.time_limit)
        config.time_limit = 1.0
        self.assertEqual(config.time_limit, 1.0)
        self.assertIsInstance(config.time_limit, float)
        config.rel_gap = 2.5
        self.assertEqual(config.rel_gap, 2.5)


class TestAutoUpdateConfig(unittest.TestCase):
    def test_interface_default_instantiation(self):
        config = AutoUpdateConfig()
        self.assertTrue(config.check_for_new_or_removed_constraints)
        self.assertTrue(config.check_for_new_or_removed_vars)
        self.assertTrue(config.check_for_new_or_removed_params)
        self.assertTrue(config.check_for_new_objective)
        self.assertTrue(config.update_constraints)
        self.assertTrue(config.update_vars)
        self.assertTrue(config.update_named_expressions)
        self.assertTrue(config.update_objective)
        self.assertTrue(config.update_objective)
        self.assertTrue(config.treat_fixed_vars_as_params)

    def test_interface_custom_instantiation(self):
        config = AutoUpdateConfig(description="A description")
        config.check_for_new_objective = False
        self.assertEqual(config._description, "A description")
        self.assertTrue(config.check_for_new_or_removed_constraints)
        self.assertFalse(config.check_for_new_objective)


class TestPersistentSolverConfig(unittest.TestCase):
    def test_interface_default_instantiation(self):
        config = PersistentSolverConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)
        self.assertTrue(config.auto_updates.check_for_new_or_removed_constraints)
        self.assertTrue(config.auto_updates.check_for_new_or_removed_vars)
        self.assertTrue(config.auto_updates.check_for_new_or_removed_params)
        self.assertTrue(config.auto_updates.check_for_new_objective)
        self.assertTrue(config.auto_updates.update_constraints)
        self.assertTrue(config.auto_updates.update_vars)
        self.assertTrue(config.auto_updates.update_named_expressions)
        self.assertTrue(config.auto_updates.update_objective)
        self.assertTrue(config.auto_updates.update_objective)
        self.assertTrue(config.auto_updates.treat_fixed_vars_as_params)

    def test_interface_custom_instantiation(self):
        config = PersistentSolverConfig(description="A description")
        config.tee = True
        config.auto_updates.check_for_new_objective = False
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertFalse(config.auto_updates.check_for_new_objective)
