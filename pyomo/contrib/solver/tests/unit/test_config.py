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

import logging
import io

import pyomo.environ as pyo
from pyomo.common import unittest
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.solver.common.config import (
    SolverConfig,
    BranchAndBoundConfig,
    AutoUpdateConfig,
    PersistentSolverConfig,
    TextIO_or_Logger,
)

ipopt, ipopt_available = attempt_import('ipopt')


class TestTextIO_or_LoggerValidator(unittest.TestCase):
    def test_booleans(self):
        ans = TextIO_or_Logger(True)
        self.assertTrue(isinstance(ans[0], io._io.TextIOWrapper))
        ans = TextIO_or_Logger(False)
        self.assertEqual(ans, [])

    def test_logger(self):
        logger = logging.getLogger('contrib.solver.config.test.1')
        ans = TextIO_or_Logger(logger)
        self.assertTrue(isinstance(ans[0], LogStream))

    @unittest.skipIf(not ipopt_available, 'ipopt is not available')
    def test_real_example(self):

        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=1, bounds=(0, None))
        m.eq = pyo.Constraint(expr=m.x[1] * m.x[2] ** 1.5 == 3)
        m.obj = pyo.Objective(expr=m.x[1] ** 2 + m.x[2] ** 2)

        solver = pyo.SolverFactory("ipopt_v2")
        with capture_output() as OUT:
            solver.solve(m, tee=True, timelimit=5)

        contents = OUT.getvalue()
        self.assertIn('EXIT: Optimal Solution Found.', contents)


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

    def test_interface_custom_instantiation(self):
        config = PersistentSolverConfig(description="A description")
        config.tee = True
        config.auto_updates.check_for_new_objective = False
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertFalse(config.auto_updates.check_for_new_objective)
