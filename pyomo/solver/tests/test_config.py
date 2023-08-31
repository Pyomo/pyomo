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
from pyomo.solver.config import SolverConfig, BranchAndBoundConfig


class TestSolverConfig(unittest.TestCase):
    def test_interface_default_instantiation(self):
        config = SolverConfig()
        self.assertEqual(config._description, None)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solution)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertFalse(config.report_timing)

    def test_interface_custom_instantiation(self):
        config = SolverConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertFalse(config.time_limit)
        config.time_limit = 1.0
        self.assertEqual(config.time_limit, 1.0)


class TestBranchAndBoundConfig(unittest.TestCase):
    def test_interface_default_instantiation(self):
        config = BranchAndBoundConfig()
        self.assertEqual(config._description, None)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solution)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertFalse(config.report_timing)
        self.assertEqual(config.rel_gap, None)
        self.assertEqual(config.abs_gap, None)
        self.assertFalse(config.relax_integrality)

    def test_interface_custom_instantiation(self):
        config = BranchAndBoundConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertFalse(config.time_limit)
        config.time_limit = 1.0
        self.assertEqual(config.time_limit, 1.0)
        config.rel_gap = 2.5
        self.assertEqual(config.rel_gap, 2.5)
