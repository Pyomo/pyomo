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
from pyomo.contrib.solver.config import SolverConfig, BranchAndBoundConfig


class TestSolverConfig(unittest.TestCase):
    def test_interface_default_instantiation(self):
        config = SolverConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solution)
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
        self.assertTrue(config.load_solution)
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
